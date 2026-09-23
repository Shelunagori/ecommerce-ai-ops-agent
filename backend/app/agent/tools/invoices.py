from langchain.tools import BaseTool, ToolRuntime, tool

from app.agent.context import AgentContext
from app.agent.tools import envelope
from app.agent.tools.envelope import Envelope
from app.agent.tools.runtime import ToolDependencies, execute, finalize
from app.agent.tools.schemas import CustomerCodeInput, InvoiceNumberInput


def build(deps: ToolDependencies) -> list[BaseTool]:
    @tool(
        "get_invoice",
        args_schema=InvoiceNumberInput,
        description=(
            "Retrieve the authoritative invoice record by exact invoice number (e.g. "
            "INV-1001): status (pending, paid, cancelled), amount, currency, issued/due/paid "
            "dates and is_overdue. is_overdue is true only for a pending invoice past its due "
            "date. Report these values as returned; do not infer payment outcomes."
        ),
    )
    def get_invoice(invoice_number: str, runtime: ToolRuntime[AgentContext]) -> Envelope:
        return execute(
            "get_invoice",
            runtime,
            deps,
            lambda q: envelope.to_jsonable(q.invoices.get_by_number(invoice_number)),
        )

    @tool(
        "get_latest_unpaid_invoice",
        args_schema=CustomerCodeInput,
        description=(
            "Get the most recently issued unpaid (status pending) invoice for a customer code, "
            "including amount, due date and is_overdue. Returns data null if the customer has "
            "no unpaid invoice. Paid and cancelled invoices are never returned."
        ),
    )
    def get_latest_unpaid_invoice(
        customer_code: str, runtime: ToolRuntime[AgentContext]
    ) -> Envelope:
        return execute(
            "get_latest_unpaid_invoice",
            runtime,
            deps,
            lambda q: envelope.to_jsonable(q.invoices.latest_unpaid_for_customer(customer_code)),
        )

    return [finalize(t, domain="invoices") for t in (get_invoice, get_latest_unpaid_invoice)]
