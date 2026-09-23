from langchain.tools import BaseTool, ToolRuntime, tool

from app.agent.context import AgentContext
from app.agent.tools import envelope
from app.agent.tools.envelope import Envelope
from app.agent.tools.runtime import ToolDependencies, execute, finalize
from app.agent.tools.schemas import SearchInput, SkuInput


def build(deps: ToolDependencies) -> list[BaseTool]:
    @tool(
        "get_product",
        args_schema=SkuInput,
        description=(
            "Get one product by exact SKU (e.g. SKU-1001): name, description, list price, "
            "currency and whether it is active. Does not report stock levels."
        ),
    )
    def get_product(sku: str, runtime: ToolRuntime[AgentContext]) -> Envelope:
        return execute(
            "get_product", runtime, deps, lambda q: envelope.to_jsonable(q.products.get_by_sku(sku))
        )

    @tool(
        "search_products",
        args_schema=SearchInput,
        description=(
            "Find products whose name contains the query text (case-insensitive). Returns a "
            "short list with SKUs; use get_product for one known SKU."
        ),
    )
    def search_products(query: str, runtime: ToolRuntime[AgentContext], limit: int = 5) -> Envelope:
        return execute(
            "search_products",
            runtime,
            deps,
            lambda q: envelope.page(q.products.search_by_name(query, limit=limit + 1), limit),
        )

    return [finalize(t, domain="products") for t in (get_product, search_products)]
