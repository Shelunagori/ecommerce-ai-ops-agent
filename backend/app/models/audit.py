"""Durable, tenant-scoped observability records (Step 10, Phase 5).

* ``agent_runs`` - one row per graph run / resume: identifiers, counts, outcome, duration.
  NEVER prompts, model responses, chain-of-thought, retrieved content, vectors or secrets.
* ``audit_events`` - the action lifecycle (requested, decided, expired, succeeded, failed,
  duplicate prevented), written in the SAME transaction as the state change it records.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin

AUDIT_EVENT_TYPES = (
    "action_requested",
    "approval_decided",
    "approval_expired",
    "action_succeeded",
    "action_failed",
    "action_duplicate_prevented",
)


class AgentRun(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "agent_runs"

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="RESTRICT"))
    kind: Mapped[str] = mapped_column(String(16))  # run | resume
    request_id: Mapped[str | None] = mapped_column(String(64))
    thread_key: Mapped[str | None] = mapped_column(String(80))  # internal cg1-<sha256>, opaque
    runner: Mapped[str] = mapped_column(String(32))
    profile: Mapped[str] = mapped_column(String(32))
    prompt_version: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    outcome: Mapped[str] = mapped_column(String(64))
    error_detail: Mapped[str | None] = mapped_column(String(64))
    model_calls: Mapped[int] = mapped_column(Integer)
    commerce_tool_count: Mapped[int] = mapped_column(Integer)
    policy_retrieval_count: Mapped[int] = mapped_column(Integer)
    retrieved_citation_count: Mapped[int] = mapped_column(Integer)
    final_citation_count: Mapped[int] = mapped_column(Integer)
    grounding_failure: Mapped[bool] = mapped_column(Boolean)
    action_request_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    action_status: Mapped[str | None] = mapped_column(String(24))
    duration_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        ForeignKeyConstraint(
            ["tenant_id", "action_request_id"],
            ["action_requests.tenant_id", "action_requests.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("kind IN ('run', 'resume')", name="kind_valid"),
        CheckConstraint("duration_ms >= 0", name="duration_non_negative"),
        Index("ix_agent_runs_tenant_id_created_at", "tenant_id", "created_at"),
        Index("ix_agent_runs_tenant_id_outcome", "tenant_id", "outcome"),
    )


class AuditEvent(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "audit_events"

    tenant_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    event_type: Mapped[str] = mapped_column(String(40))
    action_request_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    actor: Mapped[str] = mapped_column(String(128))  # "agent", "system" or a user subject
    request_id: Mapped[str | None] = mapped_column(String(64))
    tool_call_id: Mapped[str | None] = mapped_column(String(128))
    # Safe fields only: status, decision, failure_code, action_type, arguments_hash.
    details: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "action_request_id"],
            ["action_requests.tenant_id", "action_requests.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "event_type IN (" + ", ".join(f"'{e}'" for e in AUDIT_EVENT_TYPES) + ")",
            name="event_type_valid",
        ),
        Index("ix_audit_events_tenant_id_action_request_id", "tenant_id", "action_request_id"),
        Index("ix_audit_events_tenant_id_created_at", "tenant_id", "created_at"),
    )
