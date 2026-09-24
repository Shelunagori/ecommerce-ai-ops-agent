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
import re
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import psycopg
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Command
from psycopg_pool import PoolTimeout
from sqlalchemy.exc import SQLAlchemyError

from app.actions.capability import ACTION_TOOLS
from app.actions.errors import ApprovalExpiredError
from app.agent.assistant.errors import AssistantError
from app.agent.assistant.limits import AssistantLimits
from app.agent.assistant.result import (
    ActionSummary,
    AssistantResult,
    InvalidToolCallSummary,
    PolicyCitation,
    RetrievalSummary,
    ToolCallSummary,
)
from app.agent.context import AgentContext
from app.agent.graph.builder import _action_factory, build_commerce_graph, recursion_limit_for
from app.agent.graph.nodes import PolicyRetriever, approval_text
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
from app.core.request_context import request_id_var

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
        actions: Any = None,
        run_recorder: Callable[[uuid.UUID, dict[str, Any]], None] | None = None,
    ) -> None:
        """``run_recorder``: durable run records (``RunRecorder``), best effort; default off.
        ``retriever``: policy retriever (or zero-argument factory) for the RETRIEVE node;
        default: the semantic retriever over the configured embedding profile, built lazily.
        ``profile``: ``RAG_PROFILE`` (class default), ``AGENT_PROFILE`` (production entry
        points: + approval-gated actions, needs a checkpointer), ``STEP5_PARITY_PROFILE``
        (test-only). ``actions``: ``ActionService`` or factory (default built lazily)."""
        if profile.actions and checkpointer is None:
            raise ValueError("approval-gated actions require a checkpointer (pause/resume)")
        self._provider = provider
        self._tools = tuple(tools if tools is not None else build_commerce_tools())
        self._limits = limits or AssistantLimits.from_settings()
        self._checkpointed = checkpointer is not None
        self._profile = profile
        self._run_recorder = run_recorder
        self._actions = _action_factory(actions) if profile.actions else None
        self._graph = build_commerce_graph(
            provider,
            tools=self._tools,
            limits=self._limits,
            checkpointer=checkpointer,
            profile=profile,
            retriever=retriever,
            actions=self._actions,
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
        acts = tuple(ACTION_TOOLS) if self._profile.actions else ()
        return (*(t.name for t in self._tools), *extra, *acts)

    def thread_config(self, context: AgentContext, thread_id: str) -> dict[str, Any]:
        """LangGraph config for a tenant-scoped thread (raises ValueError for a bad ID)."""
        return {"configurable": {"thread_id": checkpoint_thread_key(context.tenant_id, thread_id)}}

    def run(
        self, text: str, context: AgentContext, *, thread_id: str | None = None
    ) -> AssistantResult:
        with _state_errors(), _request_scope(context):
            return self._run(text, context, thread_id=thread_id)

    def _run(
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
            if self._actions is not None:
                self._settle_pending_approval(config, context)
            graph_input = self._input(cleaned, config, context)
            if self._checkpointed:
                turn_id = graph_input["messages"][-1].id
            try:
                values = self._graph.invoke(graph_input, config, context=context)
            except GraphRecursionError:  # defensive only; app limits should stop first
                raise AssistantError("agent_limit_exceeded", detail="recursion_limit") from None
            result = self._result(values, started)
        except AssistantError as exc:
            self._enrich(exc, values)
            closed = None
            if turn_id is not None:
                closed = self._close_failed_turn(config, context, turn_id, exc)
            self._log(context, values, exc.code, started, thread_key, exc.detail, closed)
            raise
        self._log(context, values, _outcome(values), started, thread_key)
        return result

    def history(self, context: AgentContext, thread_id: str) -> list[dict[str, str]]:
        """User-visible conversation only: user messages and final assistant messages
        (answers, approval outcomes, failure markers). Never system prompts, tool calls,
        tool results, retrieved content or raw graph state."""
        with _state_errors():
            config = self._config(context, thread_id)
            values = self._graph.get_state(config).values
            if values.get("scope_digest") not in (None, scope_digest(context.tenant_id)):
                return []
            out = []
            for m in values.get("messages", []):
                if isinstance(m, HumanMessage):
                    out.append({"id": m.id or "", "role": "user", "content": str(m.content)})
                elif m.type == "ai" and not getattr(m, "tool_calls", None):
                    out.append({"id": m.id or "", "role": "assistant", "content": m.text})
            return out

    # --- Step 10: approval pause / resume ----------------------------------------------------
    def pending_approval(
        self, context: AgentContext, thread_id: str | None = None, *, thread_key: str | None = None
    ) -> dict[str, Any] | None:
        """The action this thread is paused on (safe dict), or None. An intentional
        approval interrupt is recognised by LangGraph's recorded interrupt, never by an
        open turn alone (which means a crash). ``thread_key``: the internal key stored on
        an action request (trusted server-side value)."""
        config = self._config(context, thread_id, thread_key=thread_key)
        snapshot = self._graph.get_state(config)
        if not snapshot.interrupts:
            return None
        if snapshot.values.get("scope_digest") not in (None, scope_digest(context.tenant_id)):
            return None
        return snapshot.values.get("pending_action")

    def resume(self, context: AgentContext, **kw: Any) -> AssistantResult:
        """See ``_resume``. Checkpoint-store failures surface as ``agent_state_unavailable``."""
        with _state_errors(), _request_scope(context):
            return self._resume(context, **kw)

    def _resume(
        self,
        context: AgentContext,
        *,
        thread_id: str | None = None,
        thread_key: str | None = None,
        action_id: uuid.UUID,
        decision: str,
        decided_by: str,
        expected_hash: str | None = None,
    ) -> AssistantResult:
        """Record a HUMAN decision (trusted caller) and resume the paused graph.

        Decision problems (hash mismatch, already resolved the other way, not found) raise
        the ``ActionError`` without touching the graph. An expired approval resumes the graph
        so the thread gets the deterministic 'expired, nothing changed' outcome."""
        if self._actions is None:
            raise AssistantError("agent_input_invalid", detail="actions_disabled")
        started = time.perf_counter()
        config = self._config(context, thread_id, thread_key=thread_key)
        thread_key = config.get("configurable", {}).get("thread_id")
        pending = self.pending_approval(context, thread_key=thread_key)
        if pending is None or pending.get("id") != str(action_id):
            raise AssistantError("agent_no_pending_approval")
        try:
            self._actions().decide(
                context.tenant,
                action_id,
                decision,
                decided_by=decided_by,
                expected_hash=expected_hash,
            )
        except ApprovalExpiredError:
            pass  # recorded as expired; the resume below closes the turn deterministically
        values: dict[str, Any] = {}
        try:
            values = self._resume_graph(config, context, action_id)
            result = self._result(values, started)
        except AssistantError as exc:
            self._enrich(exc, values)
            messages = self._graph.get_state(config).values.get("messages", [])
            human = [m for m in messages if isinstance(m, HumanMessage)]
            closed = self._close_failed_turn(config, context, human[-1].id, exc) if human else None
            self._log(
                context, values, exc.code, started, thread_key, exc.detail, closed, kind="resume"
            )
            raise
        self._log(context, values, _outcome(values), started, thread_key, kind="resume")
        return result

    def _resume_graph(
        self, config: dict[str, Any], context: AgentContext, action_id: uuid.UUID
    ) -> dict[str, Any]:
        try:
            return self._graph.invoke(
                Command(resume={"action_request_id": str(action_id)}), config, context=context
            )
        except GraphRecursionError:
            raise AssistantError("agent_limit_exceeded", detail="recursion_limit") from None

    def _settle_pending_approval(self, config: dict[str, Any], context: AgentContext) -> None:
        """Before a NEW user message: a paused approval must be resolved first. An expired
        (or already decided) request is closed deterministically; an open one blocks."""
        if not config.get("configurable"):
            return
        snapshot = self._graph.get_state(config)
        if not snapshot.interrupts:
            return
        pending = snapshot.values.get("pending_action")
        if pending is None:
            return
        view = self._actions().get(context.tenant, uuid.UUID(pending["id"]))  # lazy expiry
        if view.status == "pending_approval":
            raise AssistantError("agent_approval_pending")
        self._resume_graph(config, context, uuid.UUID(pending["id"]))

    @staticmethod
    def _enrich(exc: AssistantError, values: dict[str, Any]) -> None:
        exc.model_calls = exc.model_calls or values.get("model_calls", 0)
        exc.tool_calls = exc.tool_calls or _tool_summaries(values)
        exc.invalid_tool_calls = exc.invalid_tool_calls or _invalid_summaries(values)
        exc.retrievals = exc.retrievals or _retrieval_summaries(values)

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

    def _config(
        self, context: AgentContext, thread_id: str | None, *, thread_key: str | None = None
    ) -> dict[str, Any]:
        config: dict[str, Any] = {"recursion_limit": recursion_limit_for(self._limits)}
        if thread_key is not None:
            if not self._checkpointed or not _THREAD_KEY.fullmatch(thread_key):
                raise AssistantError("agent_input_invalid", detail="thread_key")
            config["configurable"] = {"thread_id": thread_key}
            return config
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
        if values.get("__interrupt__"):
            pending = values.get("pending_action")
            if pending is None:  # pragma: no cover - only the approval node interrupts
                raise RuntimeError("graph interrupted without a pending action")
            return self._build_result(values, started, approval_text(pending), pending)
        return self._build_result(values, started, values.get("answer"), values.get("action"))

    def _build_result(
        self,
        values: dict[str, Any],
        started: float,
        answer: Any,
        action: dict[str, Any] | None,
    ) -> AssistantResult:
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
            action=ActionSummary.from_view(action) if action is not None else None,
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
        kind: str = "run",
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
            "action_proposals": len(values.get("action_calls", [])),
            "outcome": outcome,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        action = values.get("action") or values.get("pending_action")
        if action:
            fields["action_id"] = action["id"]
            fields["action_type"] = action["action_type"]
            fields["action_status"] = action["status"]
        if thread_key:
            fields["thread_key"] = thread_key[:16]  # "cg1-" + 12 hex: correlatable, opaque
        if context.request_id:
            fields["request_id"] = context.request_id
        if detail:
            fields["detail"] = detail
        if failed_turn_closed is not None:
            fields["failed_turn_closed"] = failed_turn_closed
        logger.log(
            logging.INFO if outcome in ("ok", "approval_pending") else logging.WARNING,
            "graph assistant run",
            extra=fields,
        )
        if self._run_recorder is not None:
            self._run_recorder(
                context.tenant_id,
                {
                    **fields,
                    "kind": kind,
                    "profile": self._profile.name,
                    "thread_key": thread_key,
                    "request_id": context.request_id,
                    "error_detail": detail,
                    "grounding_failure": outcome == "agent_grounding_error",
                    "action_request_id": uuid.UUID(action["id"]) if action else None,
                    "action_status": action["status"] if action else None,
                },
            )


def _tool_summaries(values: dict[str, Any]) -> list[ToolCallSummary]:
    return [ToolCallSummary.model_validate(c) for c in values.get("tool_calls", [])]


def _invalid_summaries(values: dict[str, Any]) -> list[InvalidToolCallSummary]:
    return [InvalidToolCallSummary.model_validate(c) for c in values.get("invalid_tool_calls", [])]


_THREAD_KEY = re.compile(r"^cg1-[0-9a-f]{64}$")
_STATE_ERRORS: tuple[type[BaseException], ...] = (SQLAlchemyError, psycopg.Error, PoolTimeout)


@contextmanager
def _state_errors() -> Iterator[None]:
    """Checkpoint store (or DB) failures escaping LangGraph -> a stable, sanitised error.
    Business writes are never assumed: an approved action stays idempotently retryable."""
    try:
        yield
    except _STATE_ERRORS as exc:
        logger.warning("graph state store failed", extra={"error_type": type(exc).__name__})
        raise AssistantError("agent_state_unavailable") from None


@contextmanager
def _request_scope(context: AgentContext) -> Iterator[None]:
    """Expose the trusted request id to audit writers during the run (correlation)."""
    rid = context.request_id if isinstance(context, AgentContext) else None
    token = request_id_var.set(rid) if rid else None
    try:
        yield
    finally:
        if token is not None:
            request_id_var.reset(token)


def _outcome(values: dict[str, Any]) -> str:
    return "approval_pending" if values.get("__interrupt__") else "ok"


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
