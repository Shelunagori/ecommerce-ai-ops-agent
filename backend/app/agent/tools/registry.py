"""The explicit list of capabilities the future agent is permitted to use.

Nothing is auto-discovered. Every tool here is read-only and tenant-scoped through the
injected AgentContext. There is deliberately NO generic SQL / database / HTTP tool.
"""

from langchain.tools import BaseTool

from app.agent.tools import customers, invoices, orders, products, shipments
from app.agent.tools.runtime import ToolDependencies

COMMERCE_TOOL_NAMES: tuple[str, ...] = (
    "get_customer",
    "search_customers",
    "get_order",
    "list_customer_orders",
    "get_latest_customer_order",
    "get_invoice",
    "get_latest_unpaid_invoice",
    "get_shipment",
    "get_order_shipments",
    "list_delayed_shipments",
    "get_product",
    "search_products",
)


def build_commerce_tools(deps: ToolDependencies | None = None) -> list[BaseTool]:
    deps = deps or ToolDependencies()
    tools = [
        *customers.build(deps),
        *orders.build(deps),
        *invoices.build(deps),
        *shipments.build(deps),
        *products.build(deps),
    ]
    names = tuple(t.name for t in tools)
    if names != COMMERCE_TOOL_NAMES:  # guard against accidental additions/removals
        raise RuntimeError(f"tool registry mismatch: {names}")
    return tools
