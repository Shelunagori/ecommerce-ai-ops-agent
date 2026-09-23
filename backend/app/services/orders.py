import uuid

from sqlalchemy import Select, and_, select

from app.core.errors import NotFoundError
from app.models import Customer, Order, OrderItem, Product
from app.models.enums import OrderStatus
from app.schemas.commerce import OrderItemRead, OrderRead, OrderSummary
from app.services.base import DEFAULT_LIMIT, TenantScopedQueries, clamp_limit, clamp_offset
from app.services.customers import resolve_customer_id

_NEWEST_FIRST = (Order.placed_at.desc().nulls_last(), Order.order_number.desc())


class OrderQueries(TenantScopedQueries):
    def _summaries(self) -> Select:
        return (
            select(
                Order.id,
                Order.order_number,
                Customer.customer_code,
                Order.status,
                Order.currency,
                Order.total_amount,
                Order.placed_at,
            )
            .join(
                Customer,
                and_(Customer.tenant_id == Order.tenant_id, Customer.id == Order.customer_id),
            )
            .where(Order.tenant_id == self._tenant_id)
        )

    def get_by_number(self, order_number: str) -> OrderRead:
        row = self._session.execute(
            self._summaries().where(Order.order_number == order_number)
        ).first()
        if row is None:
            raise NotFoundError("order", order_number)
        items = self._items(row.id)
        return OrderRead(**_summary_fields(row), items=tuple(items))

    def list_for_customer(
        self, customer_code: str, limit: int = DEFAULT_LIMIT, offset: int = 0
    ) -> list[OrderSummary]:
        customer_id = resolve_customer_id(self._session, self._tenant_id, customer_code)
        rows = self._session.execute(
            self._summaries()
            .where(Order.customer_id == customer_id)
            .order_by(*_NEWEST_FIRST)
            .limit(clamp_limit(limit))
            .offset(clamp_offset(offset))
        )
        return [OrderSummary(**_summary_fields(r)) for r in rows]

    def latest_for_customer(self, customer_code: str) -> OrderSummary | None:
        """Most recently placed order (drafts only if nothing was ever placed)."""
        orders = self.list_for_customer(customer_code, limit=1)
        return orders[0] if orders else None

    def list_by_status(
        self, status: OrderStatus, limit: int = DEFAULT_LIMIT, offset: int = 0
    ) -> list[OrderSummary]:
        rows = self._session.execute(
            self._summaries()
            .where(Order.status == status)
            .order_by(*_NEWEST_FIRST)
            .limit(clamp_limit(limit))
            .offset(clamp_offset(offset))
        )
        return [OrderSummary(**_summary_fields(r)) for r in rows]

    def _items(self, order_id: uuid.UUID) -> list[OrderItemRead]:
        rows = self._session.execute(
            select(
                Product.sku,
                Product.name.label("product_name"),
                OrderItem.quantity,
                OrderItem.unit_price,
                OrderItem.line_total,
            )
            .join(
                Product,
                and_(Product.tenant_id == OrderItem.tenant_id, Product.id == OrderItem.product_id),
            )
            .where(OrderItem.tenant_id == self._tenant_id, OrderItem.order_id == order_id)
            # Same product may repeat across lines: tie-break deterministically.
            .order_by(Product.sku, OrderItem.created_at, OrderItem.id)
        )
        return [OrderItemRead(**r._mapping) for r in rows]


def _summary_fields(row) -> dict:  # noqa: ANN001 - SQLAlchemy Row
    fields = dict(row._mapping)
    fields.pop("id", None)
    return fields
