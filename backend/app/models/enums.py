"""Application-level enums. Stored as VARCHAR with CHECK constraints (no native PG enums)."""

from enum import StrEnum


class CustomerStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class OrderStatus(StrEnum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class InvoiceStatus(StrEnum):
    """Lifecycle status only. "Overdue" is derived: pending AND due_at < now."""

    PENDING = "pending"
    PAID = "paid"
    CANCELLED = "cancelled"


class ShipmentStatus(StrEnum):
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELAYED = "delayed"
    DELIVERED = "delivered"
    RETURNED = "returned"


class ActionType(StrEnum):
    """Approval-gated business actions (Step 10). Deliberately narrow: no generic writes."""

    CANCEL_ORDER = "cancel_order"
    ISSUE_STORE_CREDIT = "issue_store_credit"


class ActionStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


# Order statuses from which cancel_order is allowed (re-checked at execution time).
CANCELLABLE_ORDER_STATUSES = frozenset(
    {OrderStatus.DRAFT, OrderStatus.CONFIRMED, OrderStatus.PROCESSING}
)


class DocumentType(StrEnum):
    POLICY = "policy"


def check_in(column: str, enum: type[StrEnum]) -> str:
    """SQL for a CHECK constraint restricting ``column`` to the enum's values."""
    values = ", ".join(f"'{member.value}'" for member in enum)
    return f"{column} IN ({values})"
