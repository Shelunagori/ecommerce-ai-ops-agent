import uuid

from sqlalchemy import CheckConstraint, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import CustomerStatus, check_in


class Customer(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "customers"

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="RESTRICT"))
    customer_code: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(200))
    # Deliberately not unique: shared or reused addresses exist in real data.
    email: Mapped[str] = mapped_column(String(320))
    status: Mapped[str] = mapped_column(String(16), default=CustomerStatus.ACTIVE)

    __table_args__ = (
        # Target for tenant-aware composite FKs; also serves tenant_id-leading lookups.
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "customer_code"),
        CheckConstraint(check_in("status", CustomerStatus), name="status_valid"),
        Index("ix_customers_tenant_id_name", "tenant_id", "name"),
        Index("ix_customers_tenant_id_email", "tenant_id", "email"),
    )
