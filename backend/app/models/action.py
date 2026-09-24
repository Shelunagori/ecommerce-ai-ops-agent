"""Approval-gated actions (Step 10): persistent action requests and the synthetic
store-credit ledger. Both tables are tenant-scoped with tenant-aware composite FKs.

* ``action_requests`` - one row per model-proposed action. It stores the CANONICAL
  validated arguments and their sha256 (``arguments_hash``): what a human approves is
  exactly what executes. Lifecycle: pending_approval -> approved | rejected | expired;
  approved -> executing -> succeeded | failed. No chain-of-thought is stored.
* ``store_credit_transactions`` - internal synthetic credit ledger (NOT money movement).
  ``UNIQUE (tenant_id, idempotency_key)`` makes a duplicate credit for the same request
  impossible at the database level.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin
from app.models.enums import ActionStatus, ActionType, check_in
from app.models.product import CURRENCY_CHECK, MONEY

OPEN_STATUSES_SQL = "status IN ('pending_approval', 'approved', 'executing')"


class ActionRequest(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "action_requests"

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="RESTRICT"))
    action_type: Mapped[str] = mapped_column(String(32))
    # Canonical validated arguments (JSON primitives only; money as a "12.34" string).
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB)
    arguments_hash: Mapped[str] = mapped_column(String(64))
    # Stable business target for "one open request per target", e.g. "order:ORD-1004".
    target_ref: Mapped[str] = mapped_column(String(128))
    summary: Mapped[str] = mapped_column(Text)
    # Policy citations with title/version metadata (current-run catalog entries).
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    # Safe identifiers only: runner, provider, model, prompt version.
    requested_by: Mapped[dict[str, Any]] = mapped_column(JSONB)
    thread_key: Mapped[str | None] = mapped_column(String(80))  # internal cg1-... key
    tool_call_id: Mapped[str | None] = mapped_column(String(128))
    idempotency_key: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(24), default=ActionStatus.PENDING_APPROVAL)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_by: Mapped[str | None] = mapped_column(String(128))  # verified user subject
    execution_claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    failure_code: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "idempotency_key"),
        CheckConstraint(check_in("action_type", ActionType), name="action_type_valid"),
        CheckConstraint(check_in("status", ActionStatus), name="status_valid"),
        CheckConstraint("arguments_hash ~ '^[0-9a-f]{64}$'", name="arguments_hash_format"),
        CheckConstraint("expires_at > created_at", name="expires_after_created"),
        CheckConstraint(
            "status IN ('pending_approval', 'expired') OR decided_at IS NOT NULL",
            name="decided_unless_pending",
        ),
        CheckConstraint(
            "status <> 'succeeded' OR (result IS NOT NULL AND completed_at IS NOT NULL)",
            name="succeeded_has_result",
        ),
        CheckConstraint("status <> 'failed' OR failure_code IS NOT NULL", name="failed_has_code"),
        # At most one OPEN request per business target (e.g. two pending cancellations).
        Index(
            "uq_action_requests_open_target",
            "tenant_id",
            "action_type",
            "target_ref",
            unique=True,
            postgresql_where=text(OPEN_STATUSES_SQL),
        ),
        Index("ix_action_requests_tenant_id_status", "tenant_id", "status"),
        Index("ix_action_requests_tenant_id_thread_key", "tenant_id", "thread_key"),
    )


class StoreCreditTransaction(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "store_credit_transactions"

    tenant_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    customer_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    order_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    action_request_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    currency: Mapped[str] = mapped_column(String(3))
    amount: Mapped[Decimal] = mapped_column(MONEY)
    reason: Mapped[str] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "idempotency_key"),
        UniqueConstraint("tenant_id", "action_request_id"),
        ForeignKeyConstraint(
            ["tenant_id", "customer_id"],
            ["customers.tenant_id", "customers.id"],
            ondelete="RESTRICT",
        ),
        # MATCH SIMPLE: a NULL order_id (credit without an order) is allowed.
        ForeignKeyConstraint(
            ["tenant_id", "order_id"], ["orders.tenant_id", "orders.id"], ondelete="RESTRICT"
        ),
        ForeignKeyConstraint(
            ["tenant_id", "action_request_id"],
            ["action_requests.tenant_id", "action_requests.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("amount > 0", name="amount_positive"),
        CheckConstraint(CURRENCY_CHECK, name="currency_iso4217"),
        Index("ix_store_credit_transactions_tenant_id_customer_id", "tenant_id", "customer_id"),
    )
