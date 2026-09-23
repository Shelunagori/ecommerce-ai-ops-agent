"""Tenant-scoped, read-only query layer over the structured ecommerce data.

Callable directly (no HTTP) - the intended entry point for future agent tools::

    with read_only_session() as session:
        q = CommerceQueries(session, TenantContext(tenant_id))
        order = q.orders.get_by_number("ORD-1001")
"""

from sqlalchemy.orm import Session

from app.core.tenant import TenantContext
from app.services.base import Clock, utc_now
from app.services.customers import CustomerQueries
from app.services.invoices import InvoiceQueries
from app.services.orders import OrderQueries
from app.services.products import ProductQueries
from app.services.shipments import ShipmentQueries
from app.services.summary import SummaryQueries


class CommerceQueries:
    """Facade bundling every query class for one session + tenant."""

    def __init__(self, session: Session, tenant: TenantContext, clock: Clock = utc_now) -> None:
        self.tenant = tenant
        self.customers = CustomerQueries(session, tenant, clock)
        self.products = ProductQueries(session, tenant, clock)
        self.orders = OrderQueries(session, tenant, clock)
        self.invoices = InvoiceQueries(session, tenant, clock)
        self.shipments = ShipmentQueries(session, tenant, clock)
        self.summary = SummaryQueries(session, tenant, clock)


__all__ = [
    "CommerceQueries",
    "CustomerQueries",
    "InvoiceQueries",
    "OrderQueries",
    "ProductQueries",
    "ShipmentQueries",
    "SummaryQueries",
]
