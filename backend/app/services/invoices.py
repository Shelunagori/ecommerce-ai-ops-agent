from datetime import datetime

from sqlalchemy import Select, and_, func, select

from app.core.errors import NotFoundError
from app.models import Customer, Invoice, Order
from app.models.enums import InvoiceStatus
from app.schemas.commerce import InvoiceRead
from app.services.base import DEFAULT_LIMIT, TenantScopedQueries, clamp_limit, clamp_offset
from app.services.customers import resolve_customer_id


def is_overdue(status: str, due_at: datetime, now: datetime) -> bool:
    """Overdue is derived, never stored: pending AND past its due date."""
    return status == InvoiceStatus.PENDING and due_at < now


class InvoiceQueries(TenantScopedQueries):
    """ "Unpaid" means lifecycle status ``pending`` (whether or not it is overdue yet)."""

    def _base(self) -> Select:
        return (
            select(
                Invoice.invoice_number,
                Order.order_number,
                Customer.customer_code,
                Invoice.status,
                Invoice.currency,
                Invoice.amount,
                Invoice.issued_at,
                Invoice.due_at,
                Invoice.paid_at,
            )
            .join(Order, and_(Order.tenant_id == Invoice.tenant_id, Order.id == Invoice.order_id))
            .join(
                Customer,
                and_(Customer.tenant_id == Order.tenant_id, Customer.id == Order.customer_id),
            )
            .where(Invoice.tenant_id == self._tenant_id)
        )

    def _read(self, stmt: Select) -> list[InvoiceRead]:
        now = self._clock()
        return [
            InvoiceRead(**r._mapping, is_overdue=is_overdue(r.status, r.due_at, now))
            for r in self._session.execute(stmt)
        ]

    def get_by_number(self, invoice_number: str) -> InvoiceRead:
        found = self._read(self._base().where(Invoice.invoice_number == invoice_number))
        if not found:
            raise NotFoundError("invoice", invoice_number)
        return found[0]

    def list_for_customer(
        self, customer_code: str, limit: int = DEFAULT_LIMIT, offset: int = 0
    ) -> list[InvoiceRead]:
        customer_id = resolve_customer_id(self._session, self._tenant_id, customer_code)
        return self._read(
            self._base()
            .where(Order.customer_id == customer_id)
            .order_by(Invoice.issued_at.desc(), Invoice.invoice_number.desc())
            .limit(clamp_limit(limit))
            .offset(clamp_offset(offset))
        )

    def list_unpaid_for_customer(
        self, customer_code: str, limit: int = DEFAULT_LIMIT, offset: int = 0
    ) -> list[InvoiceRead]:
        """Pending invoices for the customer, most urgent (earliest due) first."""
        customer_id = resolve_customer_id(self._session, self._tenant_id, customer_code)
        return self._read(
            self._base()
            .where(Order.customer_id == customer_id, Invoice.status == InvoiceStatus.PENDING)
            .order_by(Invoice.due_at, Invoice.invoice_number)
            .limit(clamp_limit(limit))
            .offset(clamp_offset(offset))
        )

    def latest_unpaid_for_customer(self, customer_code: str) -> InvoiceRead | None:
        """Most recently issued pending invoice."""
        customer_id = resolve_customer_id(self._session, self._tenant_id, customer_code)
        found = self._read(
            self._base()
            .where(Order.customer_id == customer_id, Invoice.status == InvoiceStatus.PENDING)
            .order_by(Invoice.issued_at.desc(), Invoice.invoice_number.desc())
            .limit(1)
        )
        return found[0] if found else None

    def list_overdue(self, limit: int = DEFAULT_LIMIT, offset: int = 0) -> list[InvoiceRead]:
        """Pending invoices whose due date has passed (derived at query time)."""
        return self._read(
            self._base()
            .where(Invoice.status == InvoiceStatus.PENDING, Invoice.due_at < self._clock())
            .order_by(Invoice.due_at, Invoice.invoice_number)
            .limit(clamp_limit(limit))
            .offset(clamp_offset(offset))
        )

    def count_unpaid(self) -> int:
        return self._session.scalar(
            select(func.count())
            .select_from(Invoice)
            .where(Invoice.tenant_id == self._tenant_id, Invoice.status == InvoiceStatus.PENDING)
        )

    def count_overdue(self) -> int:
        return self._session.scalar(
            select(func.count())
            .select_from(Invoice)
            .where(
                Invoice.tenant_id == self._tenant_id,
                Invoice.status == InvoiceStatus.PENDING,
                Invoice.due_at < self._clock(),
            )
        )
