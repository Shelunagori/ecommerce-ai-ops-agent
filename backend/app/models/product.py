import uuid
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

MONEY = Numeric(12, 2)
CURRENCY_CHECK = "currency ~ '^[A-Z]{3}$'"


class Product(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "products"

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="RESTRICT"))
    sku: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    unit_price: Mapped[Decimal] = mapped_column(MONEY)
    currency: Mapped[str] = mapped_column(String(3))
    active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true())

    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "sku"),
        CheckConstraint("unit_price >= 0", name="unit_price_non_negative"),
        CheckConstraint(CURRENCY_CHECK, name="currency_iso4217"),
        Index("ix_products_tenant_id_name", "tenant_id", "name"),
    )
