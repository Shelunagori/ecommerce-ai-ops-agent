import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import OrderStatus, check_in
from app.models.product import CURRENCY_CHECK, MONEY


class Order(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "orders"

    tenant_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    customer_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    order_number: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default=OrderStatus.DRAFT)
    currency: Mapped[str] = mapped_column(String(3))
    # Kept equal to the sum of line totals by the writing code (no trigger); see docs.
    total_amount: Mapped[Decimal] = mapped_column(MONEY)
    placed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "order_number"),
        # Tenant-aware FK: the customer must belong to the same tenant.
        ForeignKeyConstraint(
            ["tenant_id", "customer_id"],
            ["customers.tenant_id", "customers.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(check_in("status", OrderStatus), name="status_valid"),
        CheckConstraint("total_amount >= 0", name="total_amount_non_negative"),
        CheckConstraint(CURRENCY_CHECK, name="currency_iso4217"),
        CheckConstraint("status = 'draft' OR placed_at IS NOT NULL", name="placed_at_unless_draft"),
        # Customer order history / latest order; also covers the composite FK.
        Index("ix_orders_tenant_id_customer_id_placed_at", "tenant_id", "customer_id", "placed_at"),
        Index("ix_orders_tenant_id_status", "tenant_id", "status"),
        Index("ix_orders_tenant_id_placed_at", "tenant_id", "placed_at"),
    )


class OrderItem(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "order_items"

    tenant_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    order_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    product_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price: Mapped[Decimal] = mapped_column(MONEY)
    line_total: Mapped[Decimal] = mapped_column(MONEY)

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "order_id"],
            ["orders.tenant_id", "orders.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "product_id"],
            ["products.tenant_id", "products.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint("unit_price >= 0", name="unit_price_non_negative"),
        CheckConstraint("line_total = quantity * unit_price", name="line_total_matches"),
        # No (order, product) uniqueness: the line's own id identifies it, and the same
        # product may appear on several lines (price, discount, customisation, ...).
        Index("ix_order_items_tenant_id_order_id", "tenant_id", "order_id"),
        Index("ix_order_items_tenant_id_product_id", "tenant_id", "product_id"),
    )
