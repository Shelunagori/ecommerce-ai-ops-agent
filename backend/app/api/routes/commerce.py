"""Read-only demo endpoints over the tenant-scoped query layer.

Every route requires the X-Tenant-ID demo header (see app.api.deps). No writes.
"""

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import Queries
from app.api.errors import ErrorResponse
from app.models.enums import OrderStatus, ShipmentStatus
from app.schemas.commerce import (
    CustomerRead,
    DemoSummary,
    InvoiceRead,
    OrderRead,
    OrderSummary,
    Page,
    ProductRead,
    ShipmentRead,
)
from app.services.base import DEFAULT_LIMIT, MAX_LIMIT

ERRORS = {
    400: {"model": ErrorResponse, "description": "Missing or malformed X-Tenant-ID"},
    404: {"model": ErrorResponse, "description": "Tenant or resource not found"},
}

router = APIRouter(prefix="/api", responses=ERRORS)

Limit = Annotated[int, Query(ge=1, le=MAX_LIMIT)]
Offset = Annotated[int, Query(ge=0)]
SearchTerm = Annotated[str | None, Query(min_length=1, max_length=100)]


# --- customers ---------------------------------------------------------------
@router.get("/customers", response_model=Page[CustomerRead], tags=["customers"])
def list_customers(
    q: Queries, search: SearchTerm = None, limit: Limit = DEFAULT_LIMIT, offset: Offset = 0
) -> Page[CustomerRead]:
    items = (
        q.customers.search_by_name(search, limit, offset)
        if search
        else q.customers.list_all(limit, offset)
    )
    return Page(items=items, limit=limit, offset=offset)


@router.get("/customers/{customer_code}", response_model=CustomerRead, tags=["customers"])
def get_customer(customer_code: str, q: Queries) -> CustomerRead:
    return q.customers.get_by_code(customer_code)


@router.get(
    "/customers/{customer_code}/orders", response_model=Page[OrderSummary], tags=["customers"]
)
def list_customer_orders(
    customer_code: str, q: Queries, limit: Limit = DEFAULT_LIMIT, offset: Offset = 0
) -> Page[OrderSummary]:
    items = q.orders.list_for_customer(customer_code, limit, offset)
    return Page(items=items, limit=limit, offset=offset)


@router.get(
    "/customers/{customer_code}/invoices/unpaid",
    response_model=Page[InvoiceRead],
    tags=["customers"],
)
def list_customer_unpaid_invoices(
    customer_code: str, q: Queries, limit: Limit = DEFAULT_LIMIT, offset: Offset = 0
) -> Page[InvoiceRead]:
    items = q.invoices.list_unpaid_for_customer(customer_code, limit, offset)
    return Page(items=items, limit=limit, offset=offset)


# --- orders ------------------------------------------------------------------
@router.get("/orders", response_model=Page[OrderSummary], tags=["orders"])
def list_orders_by_status(
    q: Queries, status: OrderStatus, limit: Limit = DEFAULT_LIMIT, offset: Offset = 0
) -> Page[OrderSummary]:
    return Page(items=q.orders.list_by_status(status, limit, offset), limit=limit, offset=offset)


@router.get("/orders/{order_number}", response_model=OrderRead, tags=["orders"])
def get_order(order_number: str, q: Queries) -> OrderRead:
    return q.orders.get_by_number(order_number)


# --- invoices ----------------------------------------------------------------
@router.get("/invoices/{invoice_number}", response_model=InvoiceRead, tags=["invoices"])
def get_invoice(invoice_number: str, q: Queries) -> InvoiceRead:
    return q.invoices.get_by_number(invoice_number)


# --- shipments ---------------------------------------------------------------
@router.get("/shipments", response_model=Page[ShipmentRead], tags=["shipments"])
def list_shipments(
    q: Queries,
    status: ShipmentStatus | None = None,
    limit: Limit = DEFAULT_LIMIT,
    offset: Offset = 0,
) -> Page[ShipmentRead]:
    return Page(items=q.shipments.list_all(status, limit, offset), limit=limit, offset=offset)


@router.get("/shipments/{shipment_number}", response_model=ShipmentRead, tags=["shipments"])
def get_shipment(shipment_number: str, q: Queries) -> ShipmentRead:
    return q.shipments.get_by_number(shipment_number)


# --- products ----------------------------------------------------------------
@router.get("/products", response_model=Page[ProductRead], tags=["products"])
def list_products(
    q: Queries, search: SearchTerm = None, limit: Limit = DEFAULT_LIMIT, offset: Offset = 0
) -> Page[ProductRead]:
    items = (
        q.products.search_by_name(search, limit, offset)
        if search
        else q.products.list_all(limit, offset)
    )
    return Page(items=items, limit=limit, offset=offset)


@router.get("/products/{sku}", response_model=ProductRead, tags=["products"])
def get_product(sku: str, q: Queries) -> ProductRead:
    return q.products.get_by_sku(sku)


# --- demo --------------------------------------------------------------------
@router.get("/demo/summary", response_model=DemoSummary, tags=["demo"])
def demo_summary(q: Queries) -> DemoSummary:
    return q.summary.demo_summary()
