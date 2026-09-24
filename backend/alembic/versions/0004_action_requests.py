"""Approval-gated actions (Step 10): action requests and the synthetic store-credit ledger.

* ``action_requests``: model-proposed actions with canonical arguments + sha256, lifecycle
  status, expiry, decision/execution metadata; tenant FK; ``UNIQUE (tenant_id,
  idempotency_key)``; partial unique index = at most one OPEN request per target.
* ``store_credit_transactions``: synthetic internal credit ledger (no real money);
  tenant-aware composite FKs to customers, orders (nullable) and action_requests;
  ``UNIQUE (tenant_id, idempotency_key)`` and ``UNIQUE (tenant_id, action_request_id)``
  make a duplicate credit for one request impossible.

* Relaxes ``ck_orders_placed_at_unless_draft`` to ``status IN ('draft', 'cancelled') OR
  placed_at IS NOT NULL``: cancel_order may cancel a DRAFT, which was never placed.

Downgrade drops both tables (ledger first) and restores the original order constraint as
``NOT VALID`` (existing cancelled drafts are kept; new rows are checked again).

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-24 10:15:14.182773

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(op.f("ck_orders_placed_at_unless_draft"), "orders", type_="check")
    op.create_check_constraint(
        op.f("ck_orders_placed_at_unless_draft"),
        "orders",
        "status IN ('draft', 'cancelled') OR placed_at IS NOT NULL",
    )
    op.create_table(
        "action_requests",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("action_type", sa.String(length=32), nullable=False),
        sa.Column("arguments", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("arguments_hash", sa.String(length=64), nullable=False),
        sa.Column("target_ref", sa.String(length=128), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("requested_by", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("thread_key", sa.String(length=80), nullable=True),
        sa.Column("tool_call_id", sa.String(length=128), nullable=True),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by", sa.String(length=128), nullable=True),
        sa.Column("execution_claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.CheckConstraint(
            "action_type IN ('cancel_order', 'issue_store_credit')",
            name=op.f("ck_action_requests_action_type_valid"),
        ),
        sa.CheckConstraint(
            "arguments_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_action_requests_arguments_hash_format"),
        ),
        sa.CheckConstraint(
            "status <> 'failed' OR failure_code IS NOT NULL",
            name=op.f("ck_action_requests_failed_has_code"),
        ),
        sa.CheckConstraint(
            "status <> 'succeeded' OR (result IS NOT NULL AND completed_at IS NOT NULL)",
            name=op.f("ck_action_requests_succeeded_has_result"),
        ),
        sa.CheckConstraint(
            "status IN ('pending_approval', 'approved', 'rejected', 'executing', "
            "'succeeded', 'failed', 'expired')",
            name=op.f("ck_action_requests_status_valid"),
        ),
        sa.CheckConstraint(
            "status IN ('pending_approval', 'expired') OR decided_at IS NOT NULL",
            name=op.f("ck_action_requests_decided_unless_pending"),
        ),
        sa.CheckConstraint(
            "expires_at > created_at", name=op.f("ck_action_requests_expires_after_created")
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_action_requests_tenant_id_tenants"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_action_requests")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_action_requests_tenant_id_id")),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name=op.f("uq_action_requests_tenant_id_idempotency_key"),
        ),
    )
    op.create_index(
        "ix_action_requests_tenant_id_status",
        "action_requests",
        ["tenant_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_action_requests_tenant_id_thread_key",
        "action_requests",
        ["tenant_id", "thread_key"],
        unique=False,
    )
    op.create_index(
        "uq_action_requests_open_target",
        "action_requests",
        ["tenant_id", "action_type", "target_ref"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending_approval', 'approved', 'executing')"),
    )
    op.create_table(
        "store_credit_transactions",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("customer_id", sa.Uuid(), nullable=False),
        sa.Column("order_id", sa.Uuid(), nullable=True),
        sa.Column("action_request_id", sa.Uuid(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("amount", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "currency ~ '^[A-Z]{3}$'", name=op.f("ck_store_credit_transactions_currency_iso4217")
        ),
        sa.CheckConstraint("amount > 0", name=op.f("ck_store_credit_transactions_amount_positive")),
        sa.ForeignKeyConstraint(
            ["tenant_id", "action_request_id"],
            ["action_requests.tenant_id", "action_requests.id"],
            name=op.f("fk_store_credit_transactions_tenant_id_action_request_id_action_requests"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "customer_id"],
            ["customers.tenant_id", "customers.id"],
            name=op.f("fk_store_credit_transactions_tenant_id_customer_id_customers"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "order_id"],
            ["orders.tenant_id", "orders.id"],
            name=op.f("fk_store_credit_transactions_tenant_id_order_id_orders"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_store_credit_transactions")),
        sa.UniqueConstraint(
            "tenant_id",
            "action_request_id",
            name=op.f("uq_store_credit_transactions_tenant_id_action_request_id"),
        ),
        sa.UniqueConstraint(
            "tenant_id", "id", name=op.f("uq_store_credit_transactions_tenant_id_id")
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name=op.f("uq_store_credit_transactions_tenant_id_idempotency_key"),
        ),
    )
    op.create_index(
        "ix_store_credit_transactions_tenant_id_customer_id",
        "store_credit_transactions",
        ["tenant_id", "customer_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_orders_placed_at_unless_draft"), "orders", type_="check")
    op.execute(
        "ALTER TABLE orders ADD CONSTRAINT ck_orders_placed_at_unless_draft "
        "CHECK (status = 'draft' OR placed_at IS NOT NULL) NOT VALID"
    )
    op.drop_index(
        "ix_store_credit_transactions_tenant_id_customer_id", table_name="store_credit_transactions"
    )
    op.drop_table("store_credit_transactions")
    op.drop_index(
        "uq_action_requests_open_target",
        table_name="action_requests",
        postgresql_where=sa.text("status IN ('pending_approval', 'approved', 'executing')"),
    )
    op.drop_index("ix_action_requests_tenant_id_thread_key", table_name="action_requests")
    op.drop_index("ix_action_requests_tenant_id_status", table_name="action_requests")
    op.drop_table("action_requests")
