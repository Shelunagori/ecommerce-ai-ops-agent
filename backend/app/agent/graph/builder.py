"""Build the commerce StateGraph.

    START -> model -+-> tools ----> model ...      (commerce tool batch)
                    +-> retrieve -> model ...      (one policy retrieval, Step 9)
                    +-> END                        (final answer or terminal error)
    tools / retrieve -> END on a host-side failure

The RETRIEVE node exists only when the profile enables policy knowledge (production
``RAG_PROFILE``); the test-only ``STEP5_PARITY_PROFILE`` builds the Step-6 topology. A later
approval step (action request -> interrupt -> action execution) can be inserted on the
``model`` -> ``tools`` path without changing the existing nodes.
"""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from app.agent.assistant.executor import ToolExecutor
from app.agent.assistant.limits import AssistantLimits
from app.agent.context import AgentContext
from app.agent.graph.nodes import ActionNodes, GraphNodes, PolicyRetriever
from app.agent.graph.profile import RAG_PROFILE, GraphProfile
from app.agent.graph.routing import (
    APPROVAL,
    EXECUTE,
    MODEL,
    PROPOSE,
    RETRIEVE,
    TOOLS,
    route_after_model,
    route_after_propose,
    route_after_retrieve,
    route_after_tools,
)
from app.agent.graph.state import CommerceGraphState
from app.agent.llm import LLMProvider
from app.agent.tools import build_commerce_tools

GRAPH_NAME = "commerce_assistant"


def recursion_limit_for(limits: AssistantLimits) -> int:
    """Defensive LangGraph step cap. The application limits stay authoritative: a run needs
    at most ``2 * max_model_rounds - 1`` steps (model and tools/retrieve alternating)."""
    return 2 * limits.max_model_rounds + 3


def build_commerce_graph(
    provider: LLMProvider,
    *,
    tools: Sequence[BaseTool] | None = None,
    limits: AssistantLimits | None = None,
    checkpointer: Checkpointer = None,
    profile: GraphProfile = RAG_PROFILE,
    retriever: PolicyRetriever | Callable[[], PolicyRetriever] | None = None,
    actions: Any = None,
) -> CompiledStateGraph:
    """``actions``: ``ActionService`` (or zero-arg factory) for ``AGENT_PROFILE``; default:
    ``ActionService()`` on read-write units of work, built lazily."""
    executor = ToolExecutor(tools if tools is not None else build_commerce_tools())
    factory = _retriever_factory(retriever) if profile.policy_knowledge else None
    action_factory = _action_factory(actions) if profile.actions else None
    nodes = GraphNodes(
        provider,
        executor,
        limits or AssistantLimits.from_settings(),
        profile,
        factory,
        action_factory,
    )
    return _compile(nodes, profile, checkpointer)


def _compile(
    nodes: GraphNodes, profile: GraphProfile, checkpointer: Checkpointer
) -> CompiledStateGraph:
    builder = StateGraph(CommerceGraphState, context_schema=AgentContext)
    builder.add_node(MODEL, nodes.model)
    builder.add_node(TOOLS, nodes.tools)
    builder.add_edge(START, MODEL)
    if profile.actions:
        acts = ActionNodes(nodes)
        builder.add_node(RETRIEVE, nodes.retrieve)
        builder.add_node(PROPOSE, acts.propose)
        builder.add_node(APPROVAL, acts.await_approval)
        builder.add_node(EXECUTE, acts.execute)
        builder.add_conditional_edges(
            MODEL,
            route_after_model,
            {TOOLS: TOOLS, RETRIEVE: RETRIEVE, PROPOSE: PROPOSE, END: END},
        )
        builder.add_conditional_edges(RETRIEVE, route_after_retrieve, {MODEL: MODEL, END: END})
        builder.add_conditional_edges(
            PROPOSE, route_after_propose, {APPROVAL: APPROVAL, MODEL: MODEL, END: END}
        )
        builder.add_edge(APPROVAL, EXECUTE)
        builder.add_edge(EXECUTE, END)
    elif profile.policy_knowledge:
        builder.add_node(RETRIEVE, nodes.retrieve)
        builder.add_conditional_edges(
            MODEL, route_after_model, {TOOLS: TOOLS, RETRIEVE: RETRIEVE, END: END}
        )
        builder.add_conditional_edges(RETRIEVE, route_after_retrieve, {MODEL: MODEL, END: END})
    else:
        builder.add_conditional_edges(MODEL, route_after_model, {TOOLS: TOOLS, END: END})
    builder.add_conditional_edges(TOOLS, route_after_tools, {MODEL: MODEL, END: END})
    return builder.compile(checkpointer=checkpointer, name=GRAPH_NAME)


def _retriever_factory(
    retriever: PolicyRetriever | Callable[[], PolicyRetriever] | None,
) -> Callable[[], PolicyRetriever]:
    """Resolve the retriever lazily: the default (semantic retriever over the configured
    embedding profile) is built on first use, so building a graph needs no network or DB."""
    if retriever is not None and hasattr(retriever, "retrieve"):
        return lambda: retriever  # type: ignore[return-value]
    if retriever is not None:
        return retriever  # type: ignore[return-value]
    cache: list[PolicyRetriever] = []

    def default() -> PolicyRetriever:
        if not cache:
            cache.append(default_policy_retriever())
        return cache[0]

    return default


def _action_factory(actions: Any) -> Callable[[], Any]:
    if actions is not None and hasattr(actions, "propose"):
        return lambda: actions
    if actions is not None:
        return actions
    cache: list[Any] = []

    def default() -> Any:
        if not cache:
            from app.actions.service import ActionService  # noqa: PLC0415

            cache.append(ActionService())
        return cache[0]

    return default


def default_policy_retriever() -> PolicyRetriever:
    """Semantic retriever on read-only sessions with the configured embedding provider."""
    from app.db.session import read_only_session  # noqa: PLC0415
    from app.knowledge.embeddings.provider import get_embedding_provider  # noqa: PLC0415
    from app.knowledge.semantic import SemanticKnowledgeRetriever  # noqa: PLC0415

    return SemanticKnowledgeRetriever(read_only_session, get_embedding_provider())
