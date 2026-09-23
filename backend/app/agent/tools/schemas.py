"""Model-visible tool input schemas.

Every schema forbids extra fields, so a smuggled ``tenant_id`` (or anything else) is
rejected. ``runtime`` is LangChain's injected ToolRuntime: LangChain removes it from the
model-facing schema, and the validator below refuses anything that is not a real
injected ToolRuntime instance (a model cannot forge one from JSON).
"""

from typing import Annotated, Any

from langchain.tools import ToolRuntime
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from app.agent.context import AgentContext

DEFAULT_LIMIT = 5
MAX_LIMIT = 20

_REFERENCE = StringConstraints(
    strip_whitespace=True, min_length=1, max_length=32, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
)


CustomerCode = Annotated[str, _REFERENCE, Field(description="Customer code, e.g. CUS-1001.")]
OrderNumber = Annotated[str, _REFERENCE, Field(description="Order number, e.g. ORD-1001.")]
InvoiceNumber = Annotated[str, _REFERENCE, Field(description="Invoice number, e.g. INV-1001.")]
ShipmentNumber = Annotated[str, _REFERENCE, Field(description="Shipment number, e.g. SHP-1001.")]
Sku = Annotated[str, _REFERENCE, Field(description="Product SKU, e.g. SKU-1001.")]
SearchQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
    Field(description="Case-insensitive text to look for in the name."),
]
Limit = Annotated[
    int,
    Field(ge=1, le=MAX_LIMIT, description=f"Maximum results to return (1-{MAX_LIMIT})."),
]


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    runtime: ToolRuntime[AgentContext]  # injected by LangChain; hidden from the model

    @field_validator("runtime", mode="wrap")
    @classmethod
    def _must_be_injected(cls, value: Any, _handler: Any) -> ToolRuntime:
        if not isinstance(value, ToolRuntime):
            raise ValueError("runtime is injected by the host, not supplied as an argument")
        return value


class CustomerCodeInput(ToolInput):
    customer_code: CustomerCode


class OrderNumberInput(ToolInput):
    order_number: OrderNumber


class InvoiceNumberInput(ToolInput):
    invoice_number: InvoiceNumber


class ShipmentNumberInput(ToolInput):
    shipment_number: ShipmentNumber


class SkuInput(ToolInput):
    sku: Sku


class SearchInput(ToolInput):
    query: SearchQuery
    limit: Limit = DEFAULT_LIMIT


class CustomerOrdersInput(ToolInput):
    customer_code: CustomerCode
    limit: Limit = DEFAULT_LIMIT


class LimitInput(ToolInput):
    limit: Limit = DEFAULT_LIMIT
