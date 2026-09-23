from sqlalchemy import Select, and_, func, select

from app.core.errors import NotFoundError
from app.models import Customer, Order, Shipment
from app.models.enums import ShipmentStatus
from app.schemas.commerce import ShipmentRead
from app.services.base import DEFAULT_LIMIT, TenantScopedQueries, clamp_limit, clamp_offset
from app.services.customers import resolve_customer_id

_NEWEST_FIRST = (
    Shipment.shipped_at.desc().nulls_first(),  # not-yet-shipped is the most recent activity
    Shipment.shipment_number.desc(),
)


class ShipmentQueries(TenantScopedQueries):
    def _base(self) -> Select:
        return (
            select(
                Shipment.shipment_number,
                Order.order_number,
                Customer.customer_code,
                Shipment.status,
                Shipment.carrier,
                Shipment.tracking_number,
                Shipment.shipped_at,
                Shipment.expected_delivery_at,
                Shipment.delivered_at,
                Shipment.delay_reason,
            )
            .join(Order, and_(Order.tenant_id == Shipment.tenant_id, Order.id == Shipment.order_id))
            .join(
                Customer,
                and_(Customer.tenant_id == Order.tenant_id, Customer.id == Order.customer_id),
            )
            .where(Shipment.tenant_id == self._tenant_id)
        )

    def _read(self, stmt: Select) -> list[ShipmentRead]:
        return [ShipmentRead(**r._mapping) for r in self._session.execute(stmt)]

    def get_by_number(self, shipment_number: str) -> ShipmentRead:
        found = self._read(self._base().where(Shipment.shipment_number == shipment_number))
        if not found:
            raise NotFoundError("shipment", shipment_number)
        return found[0]

    def list_for_order(self, order_number: str) -> list[ShipmentRead]:
        order_id = self._session.scalar(
            select(Order.id).where(
                Order.tenant_id == self._tenant_id, Order.order_number == order_number
            )
        )
        if order_id is None:
            raise NotFoundError("order", order_number)
        return self._read(
            self._base().where(Shipment.order_id == order_id).order_by(*_NEWEST_FIRST)
        )

    def latest_for_order(self, order_number: str) -> ShipmentRead | None:
        shipments = self.list_for_order(order_number)
        return shipments[0] if shipments else None

    def latest_for_customer(self, customer_code: str) -> ShipmentRead | None:
        customer_id = resolve_customer_id(self._session, self._tenant_id, customer_code)
        found = self._read(
            self._base().where(Order.customer_id == customer_id).order_by(*_NEWEST_FIRST).limit(1)
        )
        return found[0] if found else None

    def list_all(
        self,
        status: ShipmentStatus | None = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> list[ShipmentRead]:
        stmt = self._base()
        if status is not None:
            stmt = stmt.where(Shipment.status == status)
        return self._read(
            stmt.order_by(*_NEWEST_FIRST).limit(clamp_limit(limit)).offset(clamp_offset(offset))
        )

    def list_delayed(self, limit: int = DEFAULT_LIMIT, offset: int = 0) -> list[ShipmentRead]:
        return self.list_all(ShipmentStatus.DELAYED, limit=limit, offset=offset)

    def count_by_status(self, status: ShipmentStatus) -> int:
        return self._session.scalar(
            select(func.count())
            .select_from(Shipment)
            .where(Shipment.tenant_id == self._tenant_id, Shipment.status == status)
        )
