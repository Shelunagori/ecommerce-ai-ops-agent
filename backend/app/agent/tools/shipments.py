from langchain.tools import BaseTool, ToolRuntime, tool

from app.agent.context import AgentContext
from app.agent.tools import envelope
from app.agent.tools.envelope import Envelope
from app.agent.tools.runtime import ToolDependencies, execute, finalize
from app.agent.tools.schemas import MAX_LIMIT, LimitInput, OrderNumberInput, ShipmentNumberInput


def build(deps: ToolDependencies) -> list[BaseTool]:
    @tool(
        "get_shipment",
        args_schema=ShipmentNumberInput,
        description=(
            "Get one shipment by exact shipment number (e.g. SHP-1001): status, carrier, "
            "tracking number, shipped/expected/delivered dates and delay reason if known."
        ),
    )
    def get_shipment(shipment_number: str, runtime: ToolRuntime[AgentContext]) -> Envelope:
        return execute(
            "get_shipment",
            runtime,
            deps,
            lambda q: envelope.to_jsonable(q.shipments.get_by_number(shipment_number)),
        )

    @tool(
        "get_order_shipments",
        args_schema=OrderNumberInput,
        description=(
            "List the shipments for one order number, most recent first. An order may have "
            "zero, one or several shipments."
        ),
    )
    def get_order_shipments(order_number: str, runtime: ToolRuntime[AgentContext]) -> Envelope:
        return execute(
            "get_order_shipments",
            runtime,
            deps,
            lambda q: envelope.page(
                q.shipments.list_for_order(order_number, limit=MAX_LIMIT + 1), MAX_LIMIT
            ),
        )

    @tool(
        "list_delayed_shipments",
        args_schema=LimitInput,
        description=(
            "List shipments whose status is 'delayed' as reported by the carrier or "
            "operations. delay_reason may be null when no reason has been reported yet. "
            "This is not a computed 'late' check against expected delivery dates."
        ),
    )
    def list_delayed_shipments(runtime: ToolRuntime[AgentContext], limit: int = 5) -> Envelope:
        return execute(
            "list_delayed_shipments",
            runtime,
            deps,
            lambda q: envelope.page(q.shipments.list_delayed(limit=limit + 1), limit),
        )

    return [
        finalize(t, domain="shipments")
        for t in (get_shipment, get_order_shipments, list_delayed_shipments)
    ]
