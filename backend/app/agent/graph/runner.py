"""CommerceGraphAssistant: runs the LangGraph assistant and returns the Step-5 contract.

    result = CommerceGraphAssistant(provider).run("Show me order ORD-1001", context)

* ``context`` (AgentContext) is passed as LangGraph runtime context: never checkpointed,
  never in messages, prompts or tool schemas.
* Without a checkpointer every run is one-shot. With one (e.g. ``InMemorySaver``) a
  ``thread_id`` is required; it is scoped by the trusted tenant into the internal key
  ``cg1-<sha256(tenant_id:thread_id)>``. A continued thread appends the new user message
  to the thread's history (ephemeral short-term memory for as long as the checkpointer
  lives); counters, limits and outcome fields reset for every run.
* Failed turns on a checkpointed thread are closed with a generic synthetic assistant
  message (``FAILURE_MARKER_TEXT``) so the thread never holds an unanswered user message;
  the caller still gets the original ``AssistantError``. One-shot runs are unaffected.
* Policy questions (Step 9): the RETRIEVE node runs the policy retriever with the trusted
  context; the final answer is validated against the CURRENT run's source catalog.
  ``AssistantResult.retrievals`` / ``.citations`` carry safe metadata (no query text).
* Raises the Step-5 ``AssistantError`` on failure; returns ``AssistantResult`` on success.
  Graph state is never returned to callers.
"""

import logging
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from app.agent.assistant.errors import AssistantError
from app.agent.assistant.limits import AssistantLimits
from app.agent.assistant.result import (
    AssistantResult,
    InvalidToolCallSummary,
    PolicyCitation,
    RetrievalSummary,
    ToolCallSummary,
)
from app.agent.context import AgentContext
from app.agent.graph.builder import build_commerce_graph, recursion_limit_for
from app.agent.graph.nodes import PolicyRetriever
from app.agent.graph.profile import RAG_PROFILE, GraphProfile
from app.agent.graph.routing import MODEL
from app.agent.graph.state import (
    RUN_RESET,
    checkpoint_thread_key,
    failure_marker,
    scope_digest,
    turn_is_open,
)
from app.agent.llm import LLMProvider
from app.agent.rag.capability import SEARCH_POLICY_KNOWLEDGE
from app.agent.tools import build_commerce_tools

logger = logging.getLogger("app.agent.graph")

MAX_INPUT_CHARS = 4000
RUNNER = "langgraph"


class CommerceGraphAssistant:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        tools: Sequence[BaseTool] | None = None,
        limits: AssistantLimits | None = None,
        checkpointer: Checkpointer = None,
        retriever: PolicyRetriever | Callable[[], PolicyRetriever] | None = None,
        profile: GraphProfile = RAG_PROFILE,
    ) -> None:
        """``retriever``: policy retriever (or zero-argument factory) for the RETRIEVE node;
        default: the semantic retriever over the configured embedding profile, built lazily.
        ``profile``: ``RAG_PROFILE`` (production). ``STEP5_PARITY_PROFILE`` is test-only."""
        self._provider = provider
        self._tools = tuple(tools if tools is not None else build_commerce_tools())
        self._limits = limits or AssistantLimits.from_settings()
        self._checkpointed = checkpointer is not None
        self._profile = profile
        self._graph = build_commerce_graph(
            provider,
            tools=self._tools,
            limits=self._limits,
            checkpointer=checkpointer,
            profile=profile,
            retriever=retriever,
        )

    @property
    def graph(self) -> CompiledStateGraph:
        return self._graph

    @property
    def profile(self) -> GraphProfile:
        return self._profile

    @property
    def bound_tool_names(self) -> tuple[str, ...]:
        extra = (SEARCH_POLICY_KNOWLEDGE,) if self._profile.policy_knowledge else ()
        return (*(t.name for t in self._tools), *extra)

    def thread_config(self, context: AgentContext, thread_id: str) -> dict[str, Any]:
        """LangGraph config for a tenant-scoped thread (raises ValueError for a bad ID)."""
        return {"configurable": {"thread_id": checkpoint_thread_key(context.tenant_id, thread_id)}}

    def run(
        self, text: str, context: AgentContext, *, thread_id: str | None = None
    ) -> AssistantResult:
        if not isinstance(context, AgentContext):
            raise TypeError("context must be a validated AgentContext")
        started = time.perf_counter()
        thread_key: str | None = None
        config: dict[str, Any] = {}
        turn_id: str | None = None  # id of this run's user message (checkpointed runs only)
        values: dict[str, Any] = {}
        try:
            config = self._config(context, thread_id)
            thread_key = config.get("configurable", {}).get("thread_id")
            cleaned = text.strip() if isinstance(text, str) else ""
            if not cleaned or len(cleaned) > MAX_INPUT_CHARS:
                raise AssistantError("agent_input_invalid")
            graph_input = self._input(cleaned, config, context)
            if self._checkpointed:
                turn_id = graph_input["messages"][-1].id
            try:
                values = self._graph.invoke(graph_input, config, context=context)
            except GraphRecursionError:  # defensive only; app limits should stop first
                raise AssistantError("agent_limit_exceeded", detail="recursion_limit") from None
            result = self._result(values, started)
        except AssistantError as exc:
            exc.model_calls = exc.model_calls or values.get("model_calls", 0)
            exc.tool_calls = exc.tool_calls or _tool_summaries(values)
            exc.invalid_tool_calls = exc.invalid_tool_calls or _invalid_summaries(values)
            exc.retrievals = exc.retrievals or _retrieval_summaries(values)
            closed = None
            if turn_id is not None:
                closed = self._close_failed_turn(config, context, turn_id, exc)
            self._log(context, values, exc.code, started, thread_key, exc.detail, closed)
            raise
        self._log(context, values, "ok", started, thread_key)
        return result

    def _close_failed_turn(
        self, config: dict[str, Any], context: AgentContext, turn_id: str, exc: AssistantError
    ) -> bool:
        """Append the generic failure marker after this run's user message.

        Only when this run's user message is in the checkpointed thread, the thread is bound
        to this tenant and the turn is still open. Written with ``update_state`` as a new
        checkpoint (earlier checkpoints of the run are kept); no model call, no counters.
        The real code/detail stays in graph state ``error`` and in the raised error.
        """
        try:
            current = self._graph.get_state(config).values
            messages = current.get("messages", [])
            if current.get("scope_digest") not in (None, scope_digest(context.tenant_id)):
                return False  # never write into a thread bound to another tenant
            if not any(m.id == turn_id for m in messages) or not turn_is_open(messages):
                return False
            update: dict[str, Any] = {"messages": [failure_marker()], "pending": None}
            if current.get("error") is None:  # e.g. recursion limit: record the real outcome
                update["error"] = {"code": exc.code, "message": None, "detail": exc.detail}
            self._graph.update_state(config, update, as_node=MODEL)
            return True
        except Exception:  # never mask the original AssistantError
            logger.warning("graph failed-turn marker not written", exc_info=False)
            return False

    def _config(self, context: AgentContext, thread_id: str | None) -> dict[str, Any]:
        config: dict[str, Any] = {"recursion_limit": recursion_limit_for(self._limits)}
        if not self._checkpointed:
            if thread_id is not None:
                raise AssistantError("agent_input_invalid", detail="thread_id_without_checkpointer")
            return config
        if thread_id is None:
            raise AssistantError("agent_input_invalid", detail="thread_id_required")
        try:
            config.update(self.thread_config(context, thread_id))
        except ValueError:
            raise AssistantError("agent_input_invalid", detail="thread_id") from None
        return config

    def _input(self, text: str, config: dict[str, Any], context: AgentContext) -> dict[str, Any]:
        build_messages = self._profile.prompt.build_messages
        if not self._checkpointed:
            return {**RUN_RESET, "messages": build_messages(text)}
        turn = HumanMessage(content=text, id=f"turn-{uuid.uuid4().hex}")
        current = self._graph.get_state(config).values
        history = current.get("messages", [])
        if not history:
            return {**RUN_RESET, "messages": [*build_messages(text)[:-1], turn]}
        messages: list[BaseMessage] = [turn]  # continued thread: system prompt once
        if turn_is_open(history) and current.get("scope_digest") == scope_digest(context.tenant_id):
            # A previous run died without closing its turn (e.g. process crash mid-batch):
            # close it before the new user message.
            messages.insert(0, failure_marker())
        return {**RUN_RESET, "messages": messages}

    def _result(self, values: dict[str, Any], started: float) -> AssistantResult:
        error = values.get("error")
        if error is not None:
            raise AssistantError(error["code"], error.get("message"), detail=error.get("detail"))
        answer = values.get("answer")
        if not isinstance(answer, str) or not answer:
            raise RuntimeError("graph finished without an answer or an error")  # pragma: no cover
        return AssistantResult(
            answer=answer,
            provider=self._provider.info.provider,
            model=self._provider.info.model,
            prompt_version=self._profile.prompt_version,
            model_calls=values.get("model_calls", 0),
            tool_calls=_tool_summaries(values),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            retrievals=_retrieval_summaries(values),
            citations=_final_citations(values),
        )

    def _log(
        self,
        context: AgentContext,
        values: dict[str, Any],
        outcome: str,
        started: float,
        thread_key: str | None,
        detail: str | None = None,
        failed_turn_closed: bool | None = None,
    ) -> None:
        # Identifiers and counts only: never prompts, messages, model output, tool payloads,
        # graph state or the caller's raw thread ID.
        tool_calls = values.get("tool_calls", [])
        retrievals = values.get("retrievals", [])
        model_calls = values.get("model_calls", 0)
        fields: dict[str, Any] = {
            "runner": RUNNER,
            "provider": self._provider.info.provider,
            "model": self._provider.info.model,
            "prompt_version": self._profile.prompt_version,
            "tenant_id": str(context.tenant_id),
            "checkpointed": self._checkpointed,
            "model_calls": model_calls,
            "tool_call_count": len(tool_calls),
            "tool_names": [c.get("tool") for c in tool_calls],
            "invalid_tool_calls": len(values.get("invalid_tool_calls", [])),
            "graph_steps": model_calls
            + len({c.get("round") for c in tool_calls})
            + len(retrievals),
            "commerce_tool_count": len(tool_calls),
            "policy_retrieval_count": len(retrievals),
            "retrieved_citation_count": len(
                {c for r in retrievals for c in r.get("citations", [])}
            ),
            "final_citation_count": len(values.get("citations", [])),
            "retrieved_documents": _retrieved_documents(values),
            "outcome": outcome,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if thread_key:
            fields["thread_key"] = thread_key[:16]  # "cg1-" + 12 hex: correlatable, opaque
        if context.request_id:
            fields["request_id"] = context.request_id
        if detail:
            fields["detail"] = detail
        if failed_turn_closed is not None:
            fields["failed_turn_closed"] = failed_turn_closed
        logger.log(
            logging.INFO if outcome == "ok" else logging.WARNING,
            "graph assistant run",
            extra=fields,
        )


def _tool_summaries(values: dict[str, Any]) -> list[ToolCallSummary]:
    return [ToolCallSummary.model_validate(c) for c in values.get("tool_calls", [])]


def _invalid_summaries(values: dict[str, Any]) -> list[InvalidToolCallSummary]:
    return [InvalidToolCallSummary.model_validate(c) for c in values.get("invalid_tool_calls", [])]


def _retrieval_summaries(values: dict[str, Any]) -> list[RetrievalSummary]:
    return [RetrievalSummary.model_validate(r) for r in values.get("retrievals", [])]


def _final_citations(values: dict[str, Any]) -> list[PolicyCitation]:
    """Only the citations the accepted answer used, with their catalog metadata."""
    catalog = {s["citation"]: s for s in values.get("policy_sources", [])}
    return [PolicyCitation.model_validate(catalog[c]) for c in values.get("citations", [])]


def _retrieved_documents(values: dict[str, Any]) -> list[str]:
    """Distinct ``document_key@vN`` of the current-run catalog (identifiers only)."""
    seen: dict[str, None] = {}
    for s in values.get("policy_sources", []):
        seen.setdefault(f"{s['document_key']}@v{s['version']}", None)
    return list(seen)
