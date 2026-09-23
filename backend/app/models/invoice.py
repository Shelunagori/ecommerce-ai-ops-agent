import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import InvoiceStatus, check_in
from app.models.product import CURRENCY_CHECK, MONEY


class Invoice(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "invoices"

    tenant_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    order_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    invoice_number: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default=InvoiceStatus.PENDING)
    currency: Mapped[str] = mapped_column(String(3))
    amount: Mapped[Decimal] = mapped_column(MONEY)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("tenant_id", "invoice_number"),
        ForeignKeyConstraint(
            ["tenant_id", "order_id"],
            ["orders.tenant_id", "orders.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(check_in("status", InvoiceStatus), name="status_valid"),
        CheckConstraint("amount >= 0", name="amount_non_negative"),
        CheckConstraint(CURRENCY_CHECK, name="currency_iso4217"),
        CheckConstraint("due_at >= issued_at", name="due_after_issue"),
        # paid <=> paid_at present. (Refunds/cancel-after-payment are out of scope for now.)
        CheckConstraint("(status = 'paid') = (paid_at IS NOT NULL)", name="paid_at_iff_paid"),
        # Unpaid / overdue queries: status = 'pending' AND due_at < now.
        Index("ix_invoices_tenant_id_status_due_at", "tenant_id", "status", "due_at"),
        Index("ix_invoices_tenant_id_due_at", "tenant_id", "due_at"),
        Index("ix_invoices_tenant_id_order_id", "tenant_id", "order_id"),
    )
