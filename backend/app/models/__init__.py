"""Importing this package registers every model on ``Base.metadata`` (used by Alembic)."""

from app.models.action import ActionRequest, StoreCreditTransaction
from app.models.audit import AgentRun, AuditEvent
from app.models.base import Base
from app.models.customer import Customer
from app.models.invoice import Invoice
from app.models.knowledge import KnowledgeChunk, KnowledgeChunkEmbedding, KnowledgeDocument
from app.models.membership import TenantMembership
from app.models.order import Order, OrderItem
from app.models.product import Product
from app.models.shipment import Shipment
from app.models.tenant import Tenant

__all__ = [
    "ActionRequest",
    "AgentRun",
    "AuditEvent",
    "Base",
    "Customer",
    "Invoice",
    "KnowledgeChunk",
    "KnowledgeChunkEmbedding",
    "KnowledgeDocument",
    "Order",
    "OrderItem",
    "Product",
    "Shipment",
    "StoreCreditTransaction",
    "Tenant",
    "TenantMembership",
]
