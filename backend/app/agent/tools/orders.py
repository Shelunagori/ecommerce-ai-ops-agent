from langchain.tools import BaseTool, ToolRuntime, tool

from app.agent.context import AgentContext
from app.agent.tools import envelope
from app.agent.tools.envelope import Envelope
from app.agent.tools.runtime import ToolDependencies, execute, finalize
from app.agent.tools.schemas import CustomerCodeInput, CustomerOrdersInput, OrderNumberInput


def build(deps: ToolDependencies) -> list[BaseTool]:
    @tool(
        "get_order",
        args_schema=OrderNumberInput,
        description=(
            "Get one order by exact order number (e.g. ORD-1001): status, currency, total, "
            "placed date and every order line (SKU, product name, quantity, unit price, "
            "line total). Does not include invoices or shipments."
        ),
    )
    def get_order(order_number: str, runtime: ToolRuntime[AgentContext]) -> Envelope:
        return execute(
            "get_order",
            runtime,
            deps,
            lambda q: envelope.to_jsonable(q.orders.get_by_number(order_number)),
        )

    @tool(
        "list_customer_orders",
        args_schema=CustomerOrdersInput,
        description=(
            "List a customer's orders, newest first, by customer code. Returns order "
            "summaries without line items; use get_order for the lines of one order."
        ),
    )
    def list_customer_orders(
        customer_code: str, runtime: ToolRuntime[AgentContext], limit: int = 5
    ) -> Envelope:
        return execute(
            "list_customer_orders",
            runtime,
            deps,
            lambda q: envelope.page(
                q.orders.list_for_customer(customer_code, limit=limit + 1), limit
            ),
        )

    @tool(
        "get_latest_customer_order",
        args_schema=CustomerCodeInput,
        description=(
            "Get the most recently placed order summary for a customer code. Returns "
            "data null if the customer exists but has no orders."
        ),
    )
    def get_latest_customer_order(
        customer_code: str, runtime: ToolRuntime[AgentContext]
    ) -> Envelope:
        return execute(
            "get_latest_customer_order",
            runtime,
            deps,
            lambda q: envelope.to_jsonable(q.orders.latest_for_customer(customer_code)),
        )

    return [
        finalize(t, domain="orders")
        for t in (get_order, list_customer_orders, get_latest_customer_order)
    ]
