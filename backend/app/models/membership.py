"""Tenant memberships (Phase 7): which verified user (JWT ``sub``) may act in which tenant.

Server-side authority for tenant access: a client-supplied tenant selector is honoured only
when a membership row exists. ``role``: ``member`` (chat, read) or ``approver`` (may also
approve/reject action requests).
"""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin

MEMBERSHIP_ROLES = ("member", "approver")


class TenantMembership(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "tenant_memberships"

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    user_subject: Mapped[str] = mapped_column(String(128))  # verified JWT `sub`
    role: Mapped[str] = mapped_column(String(16), default="member")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("tenant_id", "user_subject"),
        CheckConstraint("role IN ('member', 'approver')", name="role_valid"),
        CheckConstraint("length(user_subject) BETWEEN 1 AND 128", name="subject_not_empty"),
        Index("ix_tenant_memberships_user_subject", "user_subject"),
    )
