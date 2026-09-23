from sqlalchemy import select

from app.core.errors import NotFoundError
from app.models import Product
from app.schemas.commerce import ProductRead
from app.services.base import (
    DEFAULT_LIMIT,
    TenantScopedQueries,
    clamp_limit,
    clamp_offset,
    like_contains,
)


class ProductQueries(TenantScopedQueries):
    def get_by_sku(self, sku: str) -> ProductRead:
        row = self._session.scalar(
            select(Product).where(Product.tenant_id == self._tenant_id, Product.sku == sku)
        )
        if row is None:
            raise NotFoundError("product", sku)
        return ProductRead.model_validate(row)

    def search_by_name(
        self, term: str, limit: int = DEFAULT_LIMIT, offset: int = 0
    ) -> list[ProductRead]:
        rows = self._session.scalars(
            select(Product)
            .where(
                Product.tenant_id == self._tenant_id,
                Product.name.ilike(like_contains(term.strip()), escape="\\"),
            )
            .order_by(Product.name, Product.sku)
            .limit(clamp_limit(limit))
            .offset(clamp_offset(offset))
        )
        return [ProductRead.model_validate(r) for r in rows]

    def list_all(self, limit: int = DEFAULT_LIMIT, offset: int = 0) -> list[ProductRead]:
        rows = self._session.scalars(
            select(Product)
            .where(Product.tenant_id == self._tenant_id)
            .order_by(Product.sku)
            .limit(clamp_limit(limit))
            .offset(clamp_offset(offset))
        )
        return [ProductRead.model_validate(r) for r in rows]
