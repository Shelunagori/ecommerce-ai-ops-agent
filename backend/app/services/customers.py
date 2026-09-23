import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.models import Customer
from app.schemas.commerce import CustomerRead
from app.services.base import (
    DEFAULT_LIMIT,
    TenantScopedQueries,
    clamp_limit,
    clamp_offset,
    like_contains,
)


class CustomerQueries(TenantScopedQueries):
    def get_by_code(self, customer_code: str) -> CustomerRead:
        row = self._session.scalar(
            select(Customer).where(
                Customer.tenant_id == self._tenant_id,
                Customer.customer_code == customer_code,
            )
        )
        if row is None:
            raise NotFoundError("customer", customer_code)
        return CustomerRead.model_validate(row)

    def search_by_name(
        self, term: str, limit: int = DEFAULT_LIMIT, offset: int = 0
    ) -> list[CustomerRead]:
        rows = self._session.scalars(
            select(Customer)
            .where(
                Customer.tenant_id == self._tenant_id,
                Customer.name.ilike(like_contains(term.strip()), escape="\\"),
            )
            .order_by(Customer.name, Customer.customer_code)
            .limit(clamp_limit(limit))
            .offset(clamp_offset(offset))
        )
        return [CustomerRead.model_validate(r) for r in rows]

    def list_all(self, limit: int = DEFAULT_LIMIT, offset: int = 0) -> list[CustomerRead]:
        rows = self._session.scalars(
            select(Customer)
            .where(Customer.tenant_id == self._tenant_id)
            .order_by(Customer.customer_code)
            .limit(clamp_limit(limit))
            .offset(clamp_offset(offset))
        )
        return [CustomerRead.model_validate(r) for r in rows]


def resolve_customer_id(session: Session, tenant_id: uuid.UUID, customer_code: str) -> uuid.UUID:
    """Tenant-scoped customer id lookup shared by the other query classes."""
    customer_id = session.scalar(
        select(Customer.id).where(
            Customer.tenant_id == tenant_id,
            Customer.customer_code == customer_code,
        )
    )
    if customer_id is None:
        raise NotFoundError("customer", customer_code)
    return customer_id
