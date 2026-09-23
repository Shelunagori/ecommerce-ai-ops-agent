"""Read models returned by the query services and the API.

Frozen and serialisable, so future agent tools can pass them around safely after the
session is closed. Internal UUIDs and tenant ids are not exposed; human-readable
references (customer_code, order_number, ...) are the public identifiers.
Money is Decimal and serialises to a JSON string to avoid float rounding.
"""

from datetime import datetime
from decimal import Decimal
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from app.models.enums import CustomerStatus, InvoiceStatus, OrderStatus, ShipmentStatus


class ReadModel(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)


class CustomerRead(ReadModel):
    customer_code: str
    name: str
    email: str
    status: CustomerStatus
    created_at: datetime


class ProductRead(ReadModel):
    sku: str
    name: str
    description: str | None
    unit_price: Decimal
    currency: str
    active: bool


class OrderItemRead(ReadModel):
    sku: str
    product_name: str
    quantity: int
    unit_price: Decimal
    line_total: Decimal


class OrderSummary(ReadModel):
    order_number: str
    customer_code: str
    status: OrderStatus
    currency: str
    total_amount: Decimal
    placed_at: datetime | None


class OrderRead(OrderSummary):
    items: tuple[OrderItemRead, ...]


class InvoiceRead(ReadModel):
    invoice_number: str
    order_number: str
    customer_code: str
    status: InvoiceStatus
    currency: str
    amount: Decimal
    issued_at: datetime
    due_at: datetime
    paid_at: datetime | None
    # Derived at read time: status == pending AND due_at < now. Not stored.
    is_overdue: bool


class ShipmentRead(ReadModel):
    shipment_number: str
    order_number: str
    customer_code: str
    status: ShipmentStatus
    carrier: str
    tracking_number: str | None
    shipped_at: datetime | None
    expected_delivery_at: datetime | None
    delivered_at: datetime | None
    delay_reason: str | None


class TenantRead(ReadModel):
    name: str
    slug: str


class DemoSummary(ReadModel):
    tenant: TenantRead
    customers: int
    products: int
    orders: int
    unpaid_invoices: int
    overdue_invoices: int
    delayed_shipments: int


T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    limit: int
    offset: int
