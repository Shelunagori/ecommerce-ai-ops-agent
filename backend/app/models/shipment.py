import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import ShipmentStatus, check_in


class Shipment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "shipments"

    tenant_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    order_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    shipment_number: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default=ShipmentStatus.PENDING)
    carrier: Mapped[str] = mapped_column(String(64))
    tracking_number: Mapped[str | None] = mapped_column(String(64))
    shipped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expected_delivery_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Nullable on purpose: a shipment can be delayed before the carrier gives a reason.
    delay_reason: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("tenant_id", "shipment_number"),
        ForeignKeyConstraint(
            ["tenant_id", "order_id"],
            ["orders.tenant_id", "orders.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(check_in("status", ShipmentStatus), name="status_valid"),
        CheckConstraint(
            "status <> 'delivered' OR delivered_at IS NOT NULL",
            name="delivered_requires_delivered_at",
        ),
        CheckConstraint(
            "delivered_at IS NULL OR status IN ('delivered', 'returned')",
            name="delivered_at_only_when_delivered_or_returned",
        ),
        CheckConstraint(
            "delivered_at IS NULL OR shipped_at IS NULL OR delivered_at >= shipped_at",
            name="delivered_after_shipped",
        ),
        Index("ix_shipments_tenant_id_status", "tenant_id", "status"),
        Index("ix_shipments_tenant_id_order_id", "tenant_id", "order_id"),
    )
