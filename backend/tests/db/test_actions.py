"""ActionService on real PostgreSQL with COMMITTED transactions (locks, constraints,
concurrency). Only business rules are exercised here; the graph approval flow is covered in
tests/db/test_hitl_graph.py."""

import threading
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, OperationalError

from app.actions import errors as E  # noqa: N812
from app.actions.schemas import arguments_hash, canonical_arguments
from app.actions.service import ActionService
from app.models import ActionRequest, Order, StoreCreditTransaction

T0 = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
EVIDENCE = [
    {
        "citation": "policy://delayed-shipment-compensation/v1#chunk-2",
        "title": "Delayed Shipment Compensation Policy",
        "document_key": "delayed-shipment-compensation",
        "version": 1,
        "section": "Compensation",
        "effective_from": "2026-01-01",
        "effective_to": None,
    }
]
BY = {"runner": "test", "provider": "fake", "model": "scripted", "prompt_version": "t"}


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def svc(committed, clock):
    return ActionService(committed, clock=clock, max_credit=Decimal("100.00"))


def cancel(svc, tenant, number="ORD-1004", call="c1", reason="Customer asked to cancel."):
    return svc.propose(
        tenant,
        "cancel_order",
        {"order_number": number, "reason": reason},
        evidence=[],
        requested_by=BY,
        thread_key="cg1-test",
        tool_call_id=call,
    )


def credit(svc, tenant, call="k1", evidence=EVIDENCE, **args):
    base = {
        "customer_code": "CUS-1002",
        "amount": "15.00",
        "currency": "USD",
        "reason": "Delayed shipment SHP-1003.",
        "order_number": "ORD-1003",
    }
    return svc.propose(
        tenant,
        "issue_store_credit",
        {**base, **args},
        evidence=evidence,
        requested_by=BY,
        thread_key="cg1-test",
        tool_call_id=call,
    )


def order_status(committed, tenant, number):
    with committed() as s:
        return s.scalar(
            select(Order.status).where(
                Order.tenant_id == tenant.tenant_id, Order.order_number == number
            )
        )


def credits(committed):
    with committed() as s:
        return s.scalar(select(func.count()).select_from(StoreCreditTransaction))


# --- canonical arguments ----------------------------------------------------------------------
def test_canonical_arguments_and_hash_are_stable():
    a = canonical_arguments(
        "issue_store_credit",
        {
            "customer_code": "cus-1002",
            "amount": "15.5",
            "currency": "usd",
            "reason": "  late \n parcel ",
        },
    )
    assert a == {
        "customer_code": "CUS-1002",
        "amount": "15.50",
        "currency": "USD",
        "reason": "late parcel",
        "order_number": None,
    }
    assert arguments_hash("issue_store_credit", a, EVIDENCE) == arguments_hash(
        "issue_store_credit", dict(a), EVIDENCE
    )
    assert arguments_hash(
        "issue_store_credit", {**a, "amount": "15.51"}, EVIDENCE
    ) != arguments_hash("issue_store_credit", a, EVIDENCE)


@pytest.mark.parametrize(
    "bad",
    [
        {"amount": 15.0},
        {"amount": 15},
        {"amount": "-1"},
        {"amount": "0"},
        {"amount": "1e3"},
        {"amount": "15.001"},
        {"currency": "EURO"},
        {"customer_code": "x y"},
        {"reason": ""},
        {"tenant_id": "11111111-1111-4111-8111-111111111111"},
        {"reason": "x" * 301},
    ],
)
def test_invalid_credit_arguments(bad):
    base = {"customer_code": "CUS-1002", "amount": "15.00", "currency": "USD", "reason": "r"}
    with pytest.raises(E.ActionArgumentsInvalidError):
        canonical_arguments("issue_store_credit", {**base, **bad})


# --- propose ------------------------------------------------------------------------------------
def test_propose_cancel_creates_pending_request_and_writes_nothing(svc, committed, tenant_a, clock):
    v = cancel(svc, tenant_a)
    assert (v.status, v.action_type, v.arguments["order_number"]) == (
        "pending_approval",
        "cancel_order",
        "ORD-1004",
    )
    assert v.expires_at == T0 + timedelta(seconds=900)
    assert "ORD-1004" in v.summary and "processing" in v.summary
    assert order_status(committed, tenant_a, "ORD-1004") == "processing"


@pytest.mark.parametrize(
    ("number", "status"),
    [("ORD-1003", "shipped"), ("ORD-1001", "delivered"), ("ORD-1006", "cancelled")],
)
def test_cancel_rejected_for_non_cancellable_status(svc, tenant_a, number, status):
    with pytest.raises(E.OrderNotCancellableError) as exc:
        cancel(svc, tenant_a, number)
    assert exc.value.detail == status


@pytest.mark.parametrize(
    "number", ["ORD-1008", "ORD-1007", "ORD-1004"]
)  # draft, confirmed, processing
def test_cancel_allowed_statuses(svc, tenant_a, number):
    assert cancel(svc, tenant_a, number, call=number).status == "pending_approval"


def test_other_tenants_order_is_not_found(svc, tenant_b):
    with pytest.raises(E.ActionTargetNotFoundError) as exc:
        cancel(svc, tenant_b, "ORD-1007")  # exists only for Northstar
    assert exc.value.code == "order_not_found"


def test_same_proposal_is_idempotent_and_conflicting_reuse_fails(svc, tenant_a):
    first = cancel(svc, tenant_a)
    assert cancel(svc, tenant_a).id == first.id
    with pytest.raises(E.IdempotencyConflictError):
        cancel(svc, tenant_a, reason="Different reason")


def test_one_open_request_per_target(svc, tenant_a):
    cancel(svc, tenant_a, call="c1")
    with pytest.raises(E.DuplicateOpenActionError):
        cancel(svc, tenant_a, call="c2")


def test_store_credit_requires_evidence(svc, tenant_a):
    with pytest.raises(E.EvidenceRequiredError):
        credit(svc, tenant_a, evidence=[])


@pytest.mark.parametrize(
    ("args", "detail"),
    [
        (
            {"customer_code": "CUS-1001", "order_number": "ORD-1002", "amount": "60.00"},
            "exceeds_order_total",
        ),
        ({"currency": "EUR"}, "currency_mismatch"),
        ({"order_number": "ORD-1001"}, "order_customer_mismatch"),
        ({"order_number": None, "amount": "150.00"}, "exceeds_maximum"),
    ],
)
def test_store_credit_rules(svc, tenant_a, args, detail):
    with pytest.raises(E.StoreCreditRejectedError) as exc:
        credit(svc, tenant_a, **args)
    assert exc.value.detail == detail


# --- decide -----------------------------------------------------------------------------------
def test_approve_is_idempotent_and_opposite_decision_refused(svc, tenant_a):
    v = cancel(svc, tenant_a)
    assert svc.decide(tenant_a, v.id, "approve", decided_by="user-1").status == "approved"
    assert svc.decide(tenant_a, v.id, "approve", decided_by="user-1").status == "approved"
    with pytest.raises(E.ApprovalResolvedError):
        svc.decide(tenant_a, v.id, "reject", decided_by="user-1")


def test_rejected_request_never_executes(svc, committed, tenant_a):
    v = cancel(svc, tenant_a)
    svc.decide(tenant_a, v.id, "reject", decided_by="user-1")
    with pytest.raises(E.ActionNotApprovedError):
        svc.execute(tenant_a, v.id)
    assert order_status(committed, tenant_a, "ORD-1004") == "processing"


def test_expired_approval_is_persisted_and_executes_nothing(svc, committed, tenant_a, clock):
    v = cancel(svc, tenant_a)
    clock.now = T0 + timedelta(seconds=901)
    with pytest.raises(E.ApprovalExpiredError):
        svc.decide(tenant_a, v.id, "approve", decided_by="user-1")
    assert svc.get(tenant_a, v.id).status == "expired"
    with pytest.raises(E.ApprovalExpiredError):
        svc.execute(tenant_a, v.id)
    assert order_status(committed, tenant_a, "ORD-1004") == "processing"


def test_approved_but_expired_before_execution_executes_nothing(svc, committed, tenant_a, clock):
    v = cancel(svc, tenant_a)
    svc.decide(tenant_a, v.id, "approve", decided_by="user-1")
    clock.now = T0 + timedelta(seconds=901)
    with pytest.raises(E.ApprovalExpiredError):
        svc.execute(tenant_a, v.id)
    assert svc.get(tenant_a, v.id).status == "expired"
    assert order_status(committed, tenant_a, "ORD-1004") == "processing"


def test_approval_bound_to_the_hash_that_was_shown(svc, tenant_a):
    v = cancel(svc, tenant_a)
    with pytest.raises(E.ArgumentsHashMismatchError):
        svc.decide(tenant_a, v.id, "approve", decided_by="u", expected_hash="0" * 64)
    assert svc.get(tenant_a, v.id).status == "pending_approval"
    assert (
        svc.decide(tenant_a, v.id, "approve", decided_by="u", expected_hash=v.arguments_hash).status
        == "approved"
    )


def test_cross_tenant_ids_are_not_found(svc, tenant_a, tenant_b):
    v = cancel(svc, tenant_a)
    for op in (
        lambda: svc.get(tenant_b, v.id),
        lambda: svc.decide(tenant_b, v.id, "approve", decided_by="u"),
        lambda: svc.execute(tenant_b, v.id),
    ):
        with pytest.raises(E.ActionNotFoundError):
            op()
    assert svc.get(tenant_a, v.id).status == "pending_approval"


# --- execute ----------------------------------------------------------------------------------
def test_execute_before_approval_writes_nothing(svc, committed, tenant_a):
    v = cancel(svc, tenant_a)
    with pytest.raises(E.ActionNotApprovedError):
        svc.execute(tenant_a, v.id)
    assert order_status(committed, tenant_a, "ORD-1004") == "processing"


def test_cancel_executes_once_and_duplicate_returns_original(svc, committed, tenant_a):
    v = cancel(svc, tenant_a)
    svc.decide(tenant_a, v.id, "approve", decided_by="u")
    done = svc.execute(tenant_a, v.id)
    assert done.status == "succeeded" and done.result == {
        "order_number": "ORD-1004",
        "previous_status": "processing",
        "status": "cancelled",
    }
    assert order_status(committed, tenant_a, "ORD-1004") == "cancelled"
    again = svc.execute(tenant_a, v.id)
    assert again.result == done.result and again.completed_at == done.completed_at


def test_precondition_changed_between_approval_and_execution(svc, committed, tenant_a):
    v = cancel(svc, tenant_a)
    svc.decide(tenant_a, v.id, "approve", decided_by="u")
    with committed() as s:
        s.execute(
            text(
                "UPDATE orders SET status='shipped' WHERE tenant_id=:t AND order_number='ORD-1004'"
            ),
            {"t": tenant_a.tenant_id},
        )
    with pytest.raises(E.OrderNotCancellableError):
        svc.execute(tenant_a, v.id)
    after = svc.get(tenant_a, v.id)
    assert (after.status, after.failure_code) == ("failed", "order_not_cancellable")
    assert order_status(committed, tenant_a, "ORD-1004") == "shipped"


def test_store_credit_executes_once(svc, committed, tenant_a):
    v = credit(svc, tenant_a)
    svc.decide(tenant_a, v.id, "approve", decided_by="u")
    done = svc.execute(tenant_a, v.id)
    again = svc.execute(tenant_a, v.id)
    assert done.result == again.result and done.result["amount"] == "15.00"
    assert credits(committed) == 1
    with committed() as s:
        row = s.scalars(select(StoreCreditTransaction)).one()
    assert (
        row.amount == Decimal("15.00") and isinstance(row.amount, Decimal) and row.currency == "USD"
    )


def test_concurrent_duplicate_execution_creates_one_credit(svc, committed, tenant_a):
    v = credit(svc, tenant_a)
    svc.decide(tenant_a, v.id, "approve", decided_by="u")
    outcomes, barrier = [], threading.Barrier(4)

    def worker():
        barrier.wait()
        try:
            outcomes.append(svc.execute(tenant_a, v.id).status)
        except E.ActionInProgressError:
            outcomes.append("in_progress")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert credits(committed) == 1
    assert "succeeded" in outcomes and set(outcomes) <= {"succeeded", "in_progress"}
    assert svc.get(tenant_a, v.id).status == "succeeded"


def test_stale_executing_claim_is_failed_not_reexecuted(svc, committed, tenant_a, clock):
    v = credit(svc, tenant_a)
    svc.decide(tenant_a, v.id, "approve", decided_by="u")
    with committed() as s:  # simulate a crash after claim A, before apply B committed
        s.execute(
            text(
                "UPDATE action_requests SET status='executing', execution_claimed_at=:c WHERE id=:i"
            ),
            {"c": T0, "i": v.id},
        )
    with pytest.raises(E.ActionInProgressError):
        svc.execute(tenant_a, v.id)  # claim still fresh: maybe in flight
    clock.now = T0 + timedelta(seconds=121)
    with pytest.raises(E.ActionExecutionFailedError) as exc:
        svc.execute(tenant_a, v.id)
    assert exc.value.detail == "execution_interrupted"
    assert svc.get(tenant_a, v.id).failure_code == "execution_interrupted"
    assert credits(committed) == 0


def test_db_failure_during_apply_is_failed_not_success(svc, committed, tenant_a, monkeypatch):
    v = credit(svc, tenant_a)
    svc.decide(tenant_a, v.id, "approve", decided_by="u")

    def boom(_session, _row):
        raise OperationalError("INSERT", {}, Exception("connection lost"))

    monkeypatch.setattr(svc, "_apply", boom)
    with pytest.raises(E.ActionExecutionFailedError) as exc:
        svc.execute(tenant_a, v.id)
    assert "connection" not in exc.value.message
    after = svc.get(tenant_a, v.id)
    assert (after.status, after.failure_code) == ("failed", "execution_error")
    assert credits(committed) == 0


def test_tampered_arguments_after_approval_do_not_execute(svc, committed, tenant_a):
    v = credit(svc, tenant_a)
    svc.decide(tenant_a, v.id, "approve", decided_by="u")
    with committed() as s:
        s.execute(
            text(
                "UPDATE action_requests SET arguments = "
                "jsonb_set(arguments, '{amount}', '\"99.00\"') WHERE id=:i"
            ),
            {"i": v.id},
        )
    with pytest.raises(E.ArgumentsHashMismatchError):
        svc.execute(tenant_a, v.id)
    assert svc.get(tenant_a, v.id).failure_code == "arguments_hash_mismatch"
    assert credits(committed) == 0


# --- ledger constraints -------------------------------------------------------------------------
def _raw_credit(s, tenant_id, customer_id, action_id, key="k", amount="1.00"):
    s.add(
        StoreCreditTransaction(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            customer_id=customer_id,
            action_request_id=action_id,
            currency="USD",
            amount=Decimal(amount),
            reason="r",
            idempotency_key=key,
        )
    )
    s.flush()


def test_ledger_rejects_duplicate_key_non_positive_and_cross_tenant(
    svc, committed, tenant_a, tenant_b
):
    v = credit(svc, tenant_a)
    with committed() as s:
        req = s.get(ActionRequest, v.id)
        cust = s.execute(
            text("SELECT id FROM customers WHERE tenant_id=:t AND customer_code='CUS-1002'"),
            {"t": tenant_a.tenant_id},
        ).scalar()
        other = s.execute(
            text("SELECT id FROM customers WHERE tenant_id=:t LIMIT 1"), {"t": tenant_b.tenant_id}
        ).scalar()
        tenant_id = req.tenant_id
    for kwargs in ({"amount": "0.00"}, {"amount": "-5.00"}):
        with pytest.raises(IntegrityError), committed() as s:
            _raw_credit(s, tenant_id, cust, v.id, **kwargs)
    with pytest.raises(IntegrityError), committed() as s:  # customer of another tenant
        _raw_credit(s, tenant_id, other, v.id)
    with committed() as s:
        _raw_credit(s, tenant_id, cust, v.id, key="dup")
    with pytest.raises(IntegrityError), committed() as s:
        _raw_credit(s, tenant_id, cust, v.id, key="dup")


def test_cancelling_a_draft_executes(svc, committed, tenant_a):
    """A draft was never placed (placed_at NULL): cancellation must still commit."""
    v = cancel(svc, tenant_a, "ORD-1008", call="draft")
    svc.decide(tenant_a, v.id, "approve", decided_by="u")
    assert svc.execute(tenant_a, v.id).result["previous_status"] == "draft"
    assert order_status(committed, tenant_a, "ORD-1008") == "cancelled"
