"""Importing this package registers every model on ``Base.metadata`` (used by Alembic)."""

from app.models.base import Base
from app.models.customer import Customer
from app.models.invoice import Invoice
from app.models.knowledge import KnowledgeChunk, KnowledgeDocument
from app.models.order import Order, OrderItem
from app.models.product import Product
from app.models.shipment import Shipment
from app.models.tenant import Tenant

__all__ = [
    "Base",
    "Customer",
    "Invoice",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "Order",
    "OrderItem",
    "Product",
    "Shipment",
    "Tenant",
]
