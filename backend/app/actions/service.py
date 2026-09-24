"""ActionService: propose -> decide -> execute, for approval-gated business actions.

Invariants (all enforced here, server side, from trusted TenantContext only):

* ``propose`` validates canonical arguments and business preconditions, then stores a
  ``pending_approval`` request. It performs NO business write.
* ``decide`` (approve / reject) is only possible while pending and before ``expires_at``;
  an approval may be bound to the ``arguments_hash`` the human saw. Re-deciding the same
  way is idempotent; the opposite decision on a resolved request is refused.
* ``execute`` runs only an ``approved``, unexpired request whose stored arguments still
  hash to ``arguments_hash``. Two transactions:

    A (claim)  lock request row -> approved -> executing (claimed_at) -> commit
    B (apply)  lock request row + target rows -> re-check preconditions -> business write
               -> succeeded + result -> commit          (write and status are ATOMIC)

  Because the business write and ``succeeded`` commit together, a request still in
  ``executing`` provably did not write. A stale claim (older than the claim timeout, row
  lock obtainable) is marked ``failed/execution_interrupted``: never re-executed
  automatically. Executing an already ``succeeded`` request returns the original result.
  Store credits additionally carry ``UNIQUE (tenant_id, idempotency_key)``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.actions import errors as E  # noqa: N812
from app.actions.schemas import (
    arguments_hash,
    canonical_arguments,
    derive_idempotency_key,
    parse_amount,
)
from app.core.tenant import TenantContext
from app.models import ActionRequest, Customer, Order, StoreCreditTransaction
from app.models.enums import CANCELLABLE_ORDER_STATUSES, ActionStatus, ActionType, OrderStatus
from app.services.base import Clock, utc_now

logger = logging.getLogger("app.actions")

Decision = Literal["approve", "reject"]
SessionScope = Callable[[], AbstractContextManager[Session]]


@dataclass(frozen=True)
class ActionView:
    """Safe, tenant-relative projection of an action request (no tenant id)."""

    id: uuid.UUID
    action_type: str
    status: str
    arguments: dict[str, Any]
    arguments_hash: str
    summary: str
    evidence: list[dict[str, Any]]
    created_at: datetime
    expires_at: datetime
    decided_at: datetime | None
    decided_by: str | None
    completed_at: datetime | None
    result: dict[str, Any] | None
    failure_code: str | None
    thread_key: str | None = None  # internal cg1- key; never serialised to clients

    @classmethod
    def of(cls, r: ActionRequest) -> ActionView:
        return cls(
            id=r.id,
            action_type=str(r.action_type),
            status=str(r.status),
            arguments=dict(r.arguments),
            arguments_hash=r.arguments_hash,
            summary=r.summary,
            evidence=list(r.evidence or []),
            created_at=r.created_at,
            expires_at=r.expires_at,
            decided_at=r.decided_at,
            decided_by=r.decided_by,
            completed_at=r.completed_at,
            result=dict(r.result) if r.result is not None else None,
            failure_code=r.failure_code,
            thread_key=r.thread_key,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "action_type": self.action_type,
            "status": self.status,
            "arguments": self.arguments,
            "arguments_hash": self.arguments_hash,
            "summary": self.summary,
            "evidence": self.evidence,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "failure_code": self.failure_code,
        }


def _default_scope() -> SessionScope:
    from app.db.session import unit_of_work  # noqa: PLC0415 - avoid import cycle at startup

    return unit_of_work


class ActionService:
    def __init__(
        self,
        session_scope: SessionScope | None = None,
        *,
        clock: Clock = utc_now,
        approval_ttl: timedelta | None = None,
        claim_timeout: timedelta | None = None,
        max_credit: Decimal | None = None,
        audit: Callable[..., None] | None | bool = True,
    ) -> None:
        from app.core.config import get_settings  # noqa: PLC0415

        s = get_settings()
        self._scope = session_scope or _default_scope()
        self._clock = clock
        self._ttl = approval_ttl or timedelta(seconds=s.action_approval_ttl_seconds)
        self._claim_timeout = claim_timeout or timedelta(
            seconds=s.action_execution_claim_timeout_seconds
        )
        self._max_credit = (
            max_credit if max_credit is not None else Decimal(s.store_credit_max_amount)
        )
        # audit(session, tenant_id, event, row, **fields) - SAME transaction as the change.
        # True -> durable audit_events (default); False/None -> disabled (unit tests only).
        if audit is True:
            from app.observability.audit import record_action_event  # noqa: PLC0415

            audit = record_action_event
        self._audit = audit or None

    # --- helpers --------------------------------------------------------------------------
    def _emit(
        self, session: Session, tenant_id: uuid.UUID, event: str, r: ActionRequest, **kw: Any
    ) -> None:
        if self._audit is not None:
            self._audit(session, tenant_id, event, r, **kw)

    @staticmethod
    def _lock(
        session: Session, tenant_id: uuid.UUID, action_id: uuid.UUID, *, nowait: bool = False
    ) -> ActionRequest:
        # Tenant filter is part of the lookup: another tenant's id is simply "not found".
        stmt = (
            select(ActionRequest)
            .where(ActionRequest.tenant_id == tenant_id, ActionRequest.id == action_id)
            .with_for_update(nowait=nowait)
        )
        row = session.scalars(stmt).one_or_none()
        if row is None:
            raise E.ActionNotFoundError()
        return row

    @staticmethod
    def _order(session: Session, tenant_id: uuid.UUID, number: str, *, lock: bool) -> Order:
        stmt = select(Order).where(Order.tenant_id == tenant_id, Order.order_number == number)
        if lock:
            stmt = stmt.with_for_update()
        order = session.scalars(stmt).one_or_none()
        if order is None:
            raise E.ActionTargetNotFoundError("order", number)
        return order

    @staticmethod
    def _customer(session: Session, tenant_id: uuid.UUID, code: str, *, lock: bool) -> Customer:
        stmt = select(Customer).where(
            Customer.tenant_id == tenant_id, Customer.customer_code == code
        )
        if lock:
            stmt = stmt.with_for_update(read=True)
        customer = session.scalars(stmt).one_or_none()
        if customer is None:
            raise E.ActionTargetNotFoundError("customer", code)
        return customer

    # --- business validation (proposal time AND again at execution time) --------------------
    def _check_cancel(
        self, session: Session, tenant_id: uuid.UUID, args: dict[str, Any], *, lock: bool
    ) -> Order:
        order = self._order(session, tenant_id, args["order_number"], lock=lock)
        if order.status not in CANCELLABLE_ORDER_STATUSES:
            raise E.OrderNotCancellableError(order.order_number, order.status)
        return order

    def _check_credit(
        self, session: Session, tenant_id: uuid.UUID, args: dict[str, Any], *, lock: bool
    ) -> tuple[Customer, Order | None]:
        customer = self._customer(session, tenant_id, args["customer_code"], lock=lock)
        amount = parse_amount(args["amount"])
        order = None
        if args.get("order_number"):
            order = self._order(session, tenant_id, args["order_number"], lock=False)
            if order.customer_id != customer.id:
                raise E.StoreCreditRejectedError(
                    "The order does not belong to this customer.", detail="order_customer_mismatch"
                )
            if order.currency != args["currency"]:
                raise E.StoreCreditRejectedError(
                    "The credit currency must match the order currency.", detail="currency_mismatch"
                )
            if amount > order.total_amount:
                raise E.StoreCreditRejectedError(
                    "The credit may not exceed the order total.", detail="exceeds_order_total"
                )
        else:
            currency = session.scalars(
                select(Order.currency)
                .where(Order.tenant_id == tenant_id, Order.customer_id == customer.id)
                .order_by(Order.placed_at.desc().nulls_last())
                .limit(1)
            ).first()
            if currency is None:
                raise E.StoreCreditRejectedError(
                    "The credit currency cannot be verified for a customer without orders.",
                    detail="currency_unverifiable",
                )
            if currency != args["currency"]:
                raise E.StoreCreditRejectedError(
                    "The credit currency must match the customer's order currency.",
                    detail="currency_mismatch",
                )
        if amount > self._max_credit:
            raise E.StoreCreditRejectedError(
                f"A single store credit may not exceed {self._max_credit} {args['currency']}.",
                detail="exceeds_maximum",
            )
        return customer, order

    @staticmethod
    def _summary(
        action_type: str, args: dict[str, Any], order: Order | None, customer: Customer | None
    ) -> str:
        if action_type == ActionType.CANCEL_ORDER:
            status = order.status if order is not None else "unknown"
            number, reason = args["order_number"], args["reason"]
            return f"Cancel order {number} (currently {status}). Reason: {reason}"
        target = f" for order {args['order_number']}" if args.get("order_number") else ""
        name = f" ({customer.name})" if customer is not None else ""
        return (
            f"Issue store credit of {args['amount']} {args['currency']} to customer "
            f"{args['customer_code']}{name}{target}. Reason: {args['reason']}"
        )

    # --- propose ------------------------------------------------------------------------------
    def propose(
        self,
        tenant: TenantContext,
        action_type: str,
        raw_arguments: Any,
        *,
        evidence: list[dict[str, Any]],
        requested_by: dict[str, Any],
        thread_key: str | None,
        tool_call_id: str | None,
        idempotency_key: str | None = None,
    ) -> ActionView:
        args = canonical_arguments(action_type, raw_arguments)
        if action_type == ActionType.ISSUE_STORE_CREDIT and not evidence:
            raise E.EvidenceRequiredError()
        digest = arguments_hash(action_type, args, evidence)
        if idempotency_key is None:
            if not thread_key or not tool_call_id:
                raise E.ActionArgumentsInvalidError("An idempotency key is required.")
            idempotency_key = derive_idempotency_key(tenant.tenant_id, thread_key, tool_call_id)
        tenant_id = tenant.tenant_id
        now = self._clock()
        try:
            with self._scope() as session:
                existing = session.scalars(
                    select(ActionRequest).where(
                        ActionRequest.tenant_id == tenant_id,
                        ActionRequest.idempotency_key == idempotency_key,
                    )
                ).one_or_none()
                if existing is not None:
                    if existing.arguments_hash != digest or existing.action_type != action_type:
                        raise E.IdempotencyConflictError()
                    return ActionView.of(existing)  # same proposal re-submitted
                order = customer = None
                if action_type == ActionType.CANCEL_ORDER:
                    order = self._check_cancel(session, tenant_id, args, lock=False)
                    target = f"order:{args['order_number']}"
                else:
                    customer, order = self._check_credit(session, tenant_id, args, lock=False)
                    target = f"credit:{args['customer_code']}:{args.get('order_number') or '-'}"
                request = ActionRequest(
                    id=uuid.uuid4(),
                    tenant_id=tenant_id,
                    action_type=action_type,
                    arguments=args,
                    arguments_hash=digest,
                    target_ref=target,
                    summary=self._summary(action_type, args, order, customer),
                    evidence=evidence,
                    requested_by=requested_by,
                    thread_key=thread_key,
                    tool_call_id=tool_call_id,
                    idempotency_key=idempotency_key,
                    status=ActionStatus.PENDING_APPROVAL,
                    created_at=now,
                    expires_at=now + self._ttl,
                )
                session.add(request)
                session.flush()
                self._emit(session, tenant_id, "action_requested", request)
                view = ActionView.of(request)
        except IntegrityError as exc:
            if "uq_action_requests_open_target" in str(exc.orig):
                raise E.DuplicateOpenActionError() from None
            raise E.IdempotencyConflictError() from None
        logger.info(
            "action proposed",
            extra={
                "action_id": str(view.id),
                "action_type": action_type,
                "tenant_id": str(tenant_id),
            },
        )
        return view

    # --- read ---------------------------------------------------------------------------------
    def get(self, tenant: TenantContext, action_id: uuid.UUID) -> ActionView:
        with self._scope() as session:
            row = session.scalars(
                select(ActionRequest).where(
                    ActionRequest.tenant_id == tenant.tenant_id, ActionRequest.id == action_id
                )
            ).one_or_none()
            if row is None:
                raise E.ActionNotFoundError()
            view = ActionView.of(row)
        return self._with_lazy_expiry(tenant, view)

    def list_requests(
        self, tenant: TenantContext, *, status: str | None = None, limit: int = 50
    ) -> list[ActionView]:
        """Newest first, tenant-scoped; pending requests past their window show as expired."""
        with self._scope() as session:
            stmt = select(ActionRequest).where(ActionRequest.tenant_id == tenant.tenant_id)
            if status is not None:
                stmt = stmt.where(ActionRequest.status == status)
            rows = session.scalars(
                stmt.order_by(ActionRequest.created_at.desc()).limit(max(1, min(limit, 100)))
            ).all()
            views = [ActionView.of(r) for r in rows]
        return [self._with_lazy_expiry(tenant, v) for v in views]

    def events(self, tenant: TenantContext, action_id: uuid.UUID) -> list[dict[str, Any]]:
        """Safe audit trail of one request (event type, actor, time, safe details)."""
        from app.models import AuditEvent  # noqa: PLC0415

        with self._scope() as session:
            rows = session.scalars(
                select(AuditEvent)
                .where(
                    AuditEvent.tenant_id == tenant.tenant_id,
                    AuditEvent.action_request_id == action_id,
                )
                .order_by(AuditEvent.created_at)
            ).all()
            return [
                {
                    "event_type": e.event_type,
                    "actor": e.actor,
                    "at": e.created_at.isoformat(),
                    "details": dict(e.details),
                }
                for e in rows
            ]

    def _with_lazy_expiry(self, tenant: TenantContext, view: ActionView) -> ActionView:
        if view.status == ActionStatus.PENDING_APPROVAL and self._clock() >= view.expires_at:
            return self.expire(tenant, view.id)
        return view

    def expire(self, tenant: TenantContext, action_id: uuid.UUID) -> ActionView:
        """Mark a pending request expired once its window has passed (idempotent)."""
        with self._scope() as session:
            row = self._lock(session, tenant.tenant_id, action_id)
            if row.status == ActionStatus.PENDING_APPROVAL and self._clock() >= row.expires_at:
                row.status = ActionStatus.EXPIRED
                row.completed_at = self._clock()
                self._emit(session, tenant.tenant_id, "approval_expired", row)
            view = ActionView.of(row)
        return view

    # --- decide -------------------------------------------------------------------------------
    def decide(
        self,
        tenant: TenantContext,
        action_id: uuid.UUID,
        decision: Decision,
        *,
        decided_by: str,
        expected_hash: str | None = None,
    ) -> ActionView:
        if decision not in ("approve", "reject"):
            raise E.ActionArgumentsInvalidError("decision must be approve or reject.")
        target = ActionStatus.APPROVED if decision == "approve" else ActionStatus.REJECTED
        expired = False
        with self._scope() as session:
            row = self._lock(session, tenant.tenant_id, action_id)
            if expected_hash is not None and expected_hash != row.arguments_hash:
                raise E.ArgumentsHashMismatchError()
            if row.status == ActionStatus.PENDING_APPROVAL:
                if self._clock() >= row.expires_at:
                    row.status = ActionStatus.EXPIRED
                    row.completed_at = self._clock()
                    self._emit(session, tenant.tenant_id, "approval_expired", row)
                    expired = True
                else:
                    row.status = target
                    row.decided_at = self._clock()
                    row.decided_by = decided_by[:128]
                    self._emit(
                        session, tenant.tenant_id, "approval_decided", row, decision=decision
                    )
            else:
                same = (
                    decision == "approve"
                    and row.status
                    in (
                        ActionStatus.APPROVED,
                        ActionStatus.EXECUTING,
                        ActionStatus.SUCCEEDED,
                        ActionStatus.FAILED,
                    )
                ) or (decision == "reject" and row.status == ActionStatus.REJECTED)
                if row.status == ActionStatus.EXPIRED:
                    raise E.ApprovalExpiredError()
                if not same:
                    raise E.ApprovalResolvedError(detail=row.status)
            view = ActionView.of(row)
        if expired:
            raise E.ApprovalExpiredError()
        return view

    # --- execute ------------------------------------------------------------------------------
    def execute(self, tenant: TenantContext, action_id: uuid.UUID) -> ActionView:
        tenant_id = tenant.tenant_id
        claimed_at = self._claim(tenant, action_id)
        if isinstance(claimed_at, ActionView):
            return claimed_at  # already succeeded: original result, no second write
        try:
            with self._scope() as session:
                row = self._lock(session, tenant_id, action_id)
                if row.status == ActionStatus.SUCCEEDED:
                    return ActionView.of(row)
                if row.status != ActionStatus.EXECUTING or row.execution_claimed_at != claimed_at:
                    raise E.ActionInProgressError()
                if row.arguments_hash != arguments_hash(
                    row.action_type, row.arguments, row.evidence or []
                ):
                    raise E.ArgumentsHashMismatchError()
                result = self._apply(session, row)
                row.status = ActionStatus.SUCCEEDED
                row.result = result
                row.completed_at = self._clock()
                self._emit(session, tenant_id, "action_succeeded", row)
                view = ActionView.of(row)
        except (E.ActionInProgressError, E.ActionNotFoundError):
            raise
        except E.ActionError as exc:
            self._fail(tenant, action_id, claimed_at, exc.code)
            raise
        except SQLAlchemyError:
            logger.warning("action execution failed", extra={"action_id": str(action_id)})
            done = self._fail(tenant, action_id, claimed_at, "execution_error")
            if done is not None and done.status == ActionStatus.SUCCEEDED:
                return done  # the commit did land; report the truth
            raise E.ActionExecutionFailedError() from None
        logger.info("action executed", extra={"action_id": str(action_id), "outcome": "succeeded"})
        return view

    def _claim(self, tenant: TenantContext, action_id: uuid.UUID) -> datetime | ActionView:
        """Transaction A. Terminal state changes (expired / interrupted) are COMMITTED
        before the corresponding error is raised (raising inside the unit of work would
        roll them back)."""
        error: E.ActionError | None = None
        with self._scope() as session:
            row = self._lock(session, tenant.tenant_id, action_id)
            now = self._clock()
            status = row.status
            if status == ActionStatus.SUCCEEDED:
                self._emit(session, tenant.tenant_id, "action_duplicate_prevented", row)
                return ActionView.of(row)
            if (
                status in (ActionStatus.PENDING_APPROVAL, ActionStatus.APPROVED)
                and now >= row.expires_at
            ):
                row.status = ActionStatus.EXPIRED
                row.completed_at = now
                self._emit(session, tenant.tenant_id, "approval_expired", row)
                error = E.ApprovalExpiredError()
            elif status == ActionStatus.PENDING_APPROVAL:
                raise E.ActionNotApprovedError()
            elif status == ActionStatus.APPROVED:
                row.status = ActionStatus.EXECUTING
                row.execution_claimed_at = now
                return now
            elif status == ActionStatus.EXECUTING:
                claimed = row.execution_claimed_at
                if claimed is not None and now - claimed < self._claim_timeout:
                    raise E.ActionInProgressError()
                # Stale claim and we hold the row lock: the apply transaction never committed
                # (write and status are atomic). Never re-run it automatically.
                row.status = ActionStatus.FAILED
                row.failure_code = "execution_interrupted"
                row.completed_at = now
                self._emit(session, tenant.tenant_id, "action_failed", row)
                error = E.ActionExecutionFailedError(detail="execution_interrupted")
            elif status == ActionStatus.EXPIRED:
                raise E.ApprovalExpiredError()
            elif status == ActionStatus.REJECTED:
                raise E.ActionNotApprovedError(detail="rejected")
            else:  # failed
                raise E.ActionExecutionFailedError(detail=row.failure_code)
        assert error is not None
        raise error

    def _fail(
        self, tenant: TenantContext, action_id: uuid.UUID, claimed_at: datetime, code: str
    ) -> ActionView | None:
        """Mark OUR claim failed unless the apply transaction actually committed."""
        try:
            with self._scope() as session:
                row = self._lock(session, tenant.tenant_id, action_id)
                if row.status == ActionStatus.EXECUTING and row.execution_claimed_at == claimed_at:
                    row.status = ActionStatus.FAILED
                    row.failure_code = code[:64]
                    row.completed_at = self._clock()
                    self._emit(session, tenant.tenant_id, "action_failed", row)
                return ActionView.of(row)
        except (SQLAlchemyError, OperationalError):
            return None  # DB unreachable: status stays 'executing' (never 'succeeded')

    def _apply(self, session: Session, row: ActionRequest) -> dict[str, Any]:
        args = row.arguments
        if row.action_type == ActionType.CANCEL_ORDER:
            order = self._check_cancel(session, row.tenant_id, args, lock=True)  # re-check
            previous = order.status
            order.status = OrderStatus.CANCELLED
            session.flush()
            return {
                "order_number": order.order_number,
                "previous_status": str(previous),
                "status": str(order.status),
            }
        customer, order = self._check_credit(session, row.tenant_id, args, lock=True)
        credit = StoreCreditTransaction(
            id=uuid.uuid4(),
            tenant_id=row.tenant_id,
            customer_id=customer.id,
            order_id=order.id if order is not None else None,
            action_request_id=row.id,
            currency=args["currency"],
            amount=parse_amount(args["amount"]),
            reason=args["reason"],
            idempotency_key=row.idempotency_key,
        )
        session.add(credit)
        session.flush()  # UNIQUE (tenant_id, idempotency_key) fires here
        return {
            "transaction_id": str(credit.id),
            "customer_code": customer.customer_code,
            "order_number": args.get("order_number"),
            "amount": args["amount"],
            "currency": args["currency"],
        }
