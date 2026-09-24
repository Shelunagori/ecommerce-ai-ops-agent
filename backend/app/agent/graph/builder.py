"""Build the commerce StateGraph.

    START -> model -> route_after_model -> tools -> route_after_tools -> model ...
                                        \\-> END                    \\-> END (tool protocol error)

A later approval step (action request -> interrupt -> action execution) can be inserted
as extra nodes on the ``model`` -> ``tools`` path without changing the existing nodes.
"""

from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from app.agent.assistant.executor import ToolExecutor
from app.agent.assistant.limits import AssistantLimits
from app.agent.context import AgentContext
from app.agent.graph.nodes import GraphNodes
from app.agent.graph.routing import MODEL, TOOLS, route_after_model, route_after_tools
from app.agent.graph.state import CommerceGraphState
from app.agent.llm import LLMProvider
from app.agent.tools import build_commerce_tools

GRAPH_NAME = "commerce_assistant"


def recursion_limit_for(limits: AssistantLimits) -> int:
    """Defensive LangGraph step cap. The application limits stay authoritative: a run needs
    at most ``2 * max_model_rounds - 1`` steps (model/tools alternating)."""
    return 2 * limits.max_model_rounds + 3


def build_commerce_graph(
    provider: LLMProvider,
    *,
    tools: Sequence[BaseTool] | None = None,
    limits: AssistantLimits | None = None,
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph:
    executor = ToolExecutor(tools if tools is not None else build_commerce_tools())
    nodes = GraphNodes(provider, executor, limits or AssistantLimits.from_settings())

    builder = StateGraph(CommerceGraphState, context_schema=AgentContext)
    builder.add_node(MODEL, nodes.model)
    builder.add_node(TOOLS, nodes.tools)
    builder.add_edge(START, MODEL)
    builder.add_conditional_edges(MODEL, route_after_model, {TOOLS: TOOLS, END: END})
    builder.add_conditional_edges(TOOLS, route_after_tools, {MODEL: MODEL, END: END})
    return builder.compile(checkpointer=checkpointer, name=GRAPH_NAME)
