"""Public-demo usage (live-trace follow-up): durable per-visitor message budget.

One row per anonymous visitor, keyed by a SHA-256 digest of the verified JWT subject (the
raw subject is not stored). Not tenant data: the public demo has exactly one tenant, and the
budget is about the shared hosted-model quota. Incremented atomically before a run, given
back when the run fails; rows are tiny and may be deleted at any time (resets the budget).
"""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class PublicDemoUsage(Base):
    __tablename__ = "public_demo_usage"

    subject_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    messages: Mapped[int] = mapped_column(Integer, default=0)
    first_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("messages >= 0", name="messages_not_negative"),
        CheckConstraint("length(subject_hash) = 64", name="subject_hash_sha256"),
    )
