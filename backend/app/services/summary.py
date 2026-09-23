from sqlalchemy import func, select

from app.models import Customer, Order, Product, Tenant
from app.models.enums import ShipmentStatus
from app.schemas.commerce import DemoSummary, TenantRead
from app.services.base import TenantScopedQueries
from app.services.invoices import InvoiceQueries
from app.services.shipments import ShipmentQueries


class SummaryQueries(TenantScopedQueries):
    def _count(self, model: type[Customer] | type[Order] | type[Product]) -> int:
        return self._session.scalar(
            select(func.count()).select_from(model).where(model.tenant_id == self._tenant_id)
        )

    def demo_summary(self) -> DemoSummary:
        tenant = self._session.execute(
            select(Tenant.name, Tenant.slug).where(Tenant.id == self._tenant_id)
        ).one()
        invoices = InvoiceQueries(self._session, self._tenant, self._clock)
        shipments = ShipmentQueries(self._session, self._tenant, self._clock)
        return DemoSummary(
            tenant=TenantRead(**tenant._mapping),
            customers=self._count(Customer),
            products=self._count(Product),
            orders=self._count(Order),
            unpaid_invoices=invoices.count_unpaid(),
            overdue_invoices=invoices.count_overdue(),
            delayed_shipments=shipments.count_by_status(ShipmentStatus.DELAYED),
        )
