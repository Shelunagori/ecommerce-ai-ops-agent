"""Observability and audit records (Step 10, Phase 5): ``agent_runs`` and ``audit_events``.

Tenant-scoped with tenant-aware composite FKs to ``action_requests``. Only identifiers,
counts, codes and durations are stored: no prompts, model output, chain-of-thought,
retrieved content, vectors or secrets. Downgrade drops both tables.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-24 10:49:21.221663

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("thread_key", sa.String(length=80), nullable=True),
        sa.Column("runner", sa.String(length=32), nullable=False),
        sa.Column("profile", sa.String(length=32), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("outcome", sa.String(length=64), nullable=False),
        sa.Column("error_detail", sa.String(length=64), nullable=True),
        sa.Column("model_calls", sa.Integer(), nullable=False),
        sa.Column("commerce_tool_count", sa.Integer(), nullable=False),
        sa.Column("policy_retrieval_count", sa.Integer(), nullable=False),
        sa.Column("retrieved_citation_count", sa.Integer(), nullable=False),
        sa.Column("final_citation_count", sa.Integer(), nullable=False),
        sa.Column("grounding_failure", sa.Boolean(), nullable=False),
        sa.Column("action_request_id", sa.Uuid(), nullable=True),
        sa.Column("action_status", sa.String(length=24), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("kind IN ('run', 'resume')", name=op.f("ck_agent_runs_kind_valid")),
        sa.CheckConstraint("duration_ms >= 0", name=op.f("ck_agent_runs_duration_non_negative")),
        sa.ForeignKeyConstraint(
            ["tenant_id", "action_request_id"],
            ["action_requests.tenant_id", "action_requests.id"],
            name=op.f("fk_agent_runs_tenant_id_action_request_id_action_requests"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_agent_runs_tenant_id_tenants"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_runs")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_agent_runs_tenant_id_id")),
    )
    op.create_index(
        "ix_agent_runs_tenant_id_created_at",
        "agent_runs",
        ["tenant_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_agent_runs_tenant_id_outcome", "agent_runs", ["tenant_id", "outcome"], unique=False
    )
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=40), nullable=False),
        sa.Column("action_request_id", sa.Uuid(), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("tool_call_id", sa.String(length=128), nullable=True),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "event_type IN ('action_requested', 'approval_decided', 'approval_expired', "
            "'action_succeeded', 'action_failed', 'action_duplicate_prevented')",
            name=op.f("ck_audit_events_event_type_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "action_request_id"],
            ["action_requests.tenant_id", "action_requests.id"],
            name=op.f("fk_audit_events_tenant_id_action_request_id_action_requests"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_events")),
    )
    op.create_index(
        "ix_audit_events_tenant_id_action_request_id",
        "audit_events",
        ["tenant_id", "action_request_id"],
        unique=False,
    )
    op.create_index(
        "ix_audit_events_tenant_id_created_at",
        "audit_events",
        ["tenant_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_audit_events_tenant_id_created_at", table_name="audit_events")
    op.drop_index("ix_audit_events_tenant_id_action_request_id", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index("ix_agent_runs_tenant_id_outcome", table_name="agent_runs")
    op.drop_index("ix_agent_runs_tenant_id_created_at", table_name="agent_runs")
    op.drop_table("agent_runs")
