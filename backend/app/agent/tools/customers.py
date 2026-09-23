from langchain.tools import BaseTool, ToolRuntime, tool

from app.agent.context import AgentContext
from app.agent.tools import envelope
from app.agent.tools.envelope import Envelope
from app.agent.tools.runtime import ToolDependencies, execute, finalize
from app.agent.tools.schemas import CustomerCodeInput, SearchInput


def build(deps: ToolDependencies) -> list[BaseTool]:
    @tool(
        "get_customer",
        args_schema=CustomerCodeInput,
        description=(
            "Look up one customer by exact customer code (e.g. CUS-1001). Returns name, "
            "email, status and created date. Use search_customers when you only know a name."
        ),
    )
    def get_customer(customer_code: str, runtime: ToolRuntime[AgentContext]) -> Envelope:
        return execute(
            "get_customer",
            runtime,
            deps,
            lambda q: envelope.to_jsonable(q.customers.get_by_code(customer_code)),
        )

    @tool(
        "search_customers",
        args_schema=SearchInput,
        description=(
            "Find customers whose name contains the query text (case-insensitive). Returns "
            "a short list with customer codes; use get_customer for one known code."
        ),
    )
    def search_customers(
        query: str, runtime: ToolRuntime[AgentContext], limit: int = 5
    ) -> Envelope:
        return execute(
            "search_customers",
            runtime,
            deps,
            lambda q: envelope.page(q.customers.search_by_name(query, limit=limit + 1), limit),
        )

    return [finalize(t, domain="customers") for t in (get_customer, search_customers)]
