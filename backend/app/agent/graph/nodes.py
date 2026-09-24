"""MODEL, TOOLS and RETRIEVE nodes. Each returns only a state update.

External side effects: MODEL invokes the Step-4 provider; TOOLS executes the read-only
Step-3 tools through the hardened ``ToolExecutor`` (registry allowlist, trusted runtime
injection, argument checks, sanitised summaries, call-id checks, Step-3 logging);
RETRIEVE (Step 9) runs the policy retriever with the trusted tenant context and a fixed
limit. Policy retrieval never goes through ``ToolExecutor``.
"""

import logging
import time
from collections.abc import Callable
from datetime import date
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from sqlalchemy.exc import SQLAlchemyError

from app.agent.assistant.executor import ToolExecutionError, ToolExecutor
from app.agent.assistant.limits import AssistantLimits
from app.agent.assistant.result import RetrievalSummary
from app.agent.context import AgentContext
from app.agent.graph.profile import GraphProfile
from app.agent.graph.routing import evaluate_model_turn
from app.agent.graph.state import CommerceGraphState, GraphError, scope_digest
from app.agent.llm import LLMError, LLMProvider
from app.agent.rag.capability import (
    POLICY_RETRIEVAL_LIMIT,
    SEARCH_POLICY_KNOWLEDGE,
    PolicySearchArgumentError,
    parse_policy_search_args,
    policy_search_tool,
)
from app.agent.rag.context import format_policy_results, invalid_arguments_text
from app.agent.rag.grounding import check_grounding
from app.knowledge.embeddings.errors import EmbeddingError
from app.knowledge.retrieval import RetrievalResult

logger = logging.getLogger("app.agent.graph")

THREAD_SCOPE_MESSAGE = "This conversation thread belongs to a different account."
ARTIFACT_KEY = "policy_citations"  # ToolMessage.artifact: kept in history, never sent to models
_STATUS_RANK = {"none": 0, "invalid": 1, "no_results": 2, "success": 3}


class PolicyRetriever(Protocol):
    def retrieve(
        self, query: str, context: AgentContext, *, as_of: date | None = None, limit: int = ...
    ) -> RetrievalResult: ...


class _RetrievalCheckError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


def _trusted_context(runtime: Runtime[AgentContext]) -> AgentContext:
    context = runtime.context
    if not isinstance(context, AgentContext):
        raise TypeError("graph runtime context must be a validated AgentContext")
    return context


def _error(code: str, detail: str | None = None, message: str | None = None) -> GraphError:
    return {"code": code, "message": message, "detail": detail}


def merge_status(current: str, new: str) -> str:
    """success > no_results > invalid > none (a later invalid call never hides a success)."""
    return new if _STATUS_RANK[new] >= _STATUS_RANK.get(current, 0) else current


def earlier_policy_citations(messages: list[BaseMessage]) -> set[str]:
    """Citations returned by retrieval ToolMessages anywhere in this thread's history."""
    found: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == SEARCH_POLICY_KNOWLEDGE:
            artifact = m.artifact if isinstance(m.artifact, dict) else {}
            found.update(c for c in artifact.get(ARTIFACT_KEY, []) if isinstance(c, str))
    return found


class GraphNodes:
    def __init__(
        self,
        provider: LLMProvider,
        executor: ToolExecutor,
        limits: AssistantLimits,
        profile: GraphProfile,
        retriever: Callable[[], PolicyRetriever] | None = None,
    ) -> None:
        self._provider = provider
        self._executor = executor
        self._limits = limits
        self._profile = profile
        self._retriever = retriever
        self._bound: tuple[BaseTool, ...] = executor.tools + (
            (policy_search_tool(),) if profile.policy_knowledge else ()
        )

    @property
    def bound_tools(self) -> tuple[BaseTool, ...]:
        return self._bound

    # --- MODEL ------------------------------------------------------------------------------
    def model(self, state: CommerceGraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        context = _trusted_context(runtime)
        expected = scope_digest(context.tenant_id)
        bound = state.get("scope_digest")
        if bound is not None and bound != expected:
            # Defence in depth behind the tenant-scoped checkpoint key: refused before any
            # model call, nothing appended.
            return {
                "pending": None,
                "error": _error("agent_thread_conflict", "scope_mismatch", THREAD_SCOPE_MESSAGE),
            }
        update: dict[str, Any] = {} if bound is not None else {"scope_digest": expected}

        round_no = state.get("model_calls", 0) + 1
        update["model_calls"] = round_no
        prompt = self._profile.prompt
        try:
            chat = self._provider.invoke_chat(
                state["messages"],
                tools=self._bound,
                operation=prompt.PROMPT_ID,
                prompt_version=prompt.PROMPT_VERSION,
            )
        except LLMError as exc:
            return {**update, "pending": None, "error": _error(exc.code, "llm", exc.message)}
        ai: AIMessage = chat.message  # the provider's original message, never rebuilt

        status = state.get("policy_retrieval_status", "none")
        decision = evaluate_model_turn(
            ai,
            round_no=round_no,
            seen_call_ids=state.get("seen_tool_call_ids", []),
            # every capability call counts: commerce + policy retrieval (incl. invalid ones)
            tool_calls_so_far=len(state.get("tool_calls", [])) + len(state.get("retrievals", [])),
            limits=self._limits,
            retrieval_name=SEARCH_POLICY_KNOWLEDGE if self._profile.policy_knowledge else None,
            retrieval_done=status in ("success", "no_results"),
        )
        if decision.kind in ("tools", "retrieve"):
            seen = [*state.get("seen_tool_call_ids", []), *decision.new_call_ids]
            kind = "retrieval" if decision.kind == "retrieve" else "commerce"
            return {**update, "pending": ai, "pending_kind": kind, "seen_tool_call_ids": seen}
        if decision.kind == "answer":
            if self._profile.policy_knowledge:
                sources = {s["citation"]: s for s in state.get("policy_sources", [])}
                grounding = check_grounding(
                    decision.answer or "",
                    status=status,  # type: ignore[arg-type]
                    current=sources,
                    earlier=earlier_policy_citations(state["messages"]),
                )
                if not grounding.ok:
                    # The ungrounded answer is neither returned nor kept in history.
                    return {
                        **update,
                        "pending": None,
                        "error": _error("agent_grounding_error", grounding.detail),
                    }
                update["citations"] = grounding.citations
            return {**update, "pending": None, "messages": [ai], "answer": decision.answer}
        invalid = [
            *state.get("invalid_tool_calls", []),
            *(c.model_dump(mode="json") for c in decision.invalid_calls),
        ]
        return {
            **update,
            "pending": None,
            "invalid_tool_calls": invalid,
            "error": _error(decision.error_code or "agent_protocol_error", decision.error_detail),
        }

    # --- TOOLS ------------------------------------------------------------------------------
    def tools(self, state: CommerceGraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        context = _trusted_context(runtime)
        ai = state.get("pending")
        if ai is None or state.get("pending_kind") not in (None, "commerce"):
            raise RuntimeError("tools node reached without an approved commerce batch")
        round_no = state.get("model_calls", 0)
        summaries = list(state.get("tool_calls", []))
        tool_messages = []
        for call in ai.tool_calls:  # sequential, in request order; no model call in between
            try:
                message, summary = self._executor.execute(call, context, round_no)
            except ToolExecutionError:
                return {
                    "pending": None,
                    "pending_kind": None,
                    "tool_calls": summaries,
                    "error": _error("agent_protocol_error", "tool_result"),
                }
            tool_messages.append(message)
            summaries.append(summary.model_dump(mode="json"))
        # One update: the original AIMessage, then its complete ordered ToolMessage batch.
        return {
            "messages": [ai, *tool_messages],
            "pending": None,
            "pending_kind": None,
            "tool_calls": summaries,
        }

    # --- RETRIEVE ---------------------------------------------------------------------------
    def retrieve(self, state: CommerceGraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        context = _trusted_context(runtime)  # trusted tenant: never from the model
        ai = state.get("pending")
        if ai is None or state.get("pending_kind") != "retrieval" or len(ai.tool_calls) != 1:
            raise RuntimeError("retrieve node reached without one approved retrieval call")
        call = ai.tool_calls[0]
        round_no = state.get("model_calls", 0)
        started = time.perf_counter()
        retrievals = list(state.get("retrievals", []))
        status = state.get("policy_retrieval_status", "none")

        def summary(**fields: Any) -> dict[str, Any]:
            ms = round((time.perf_counter() - started) * 1000, 1)
            return RetrievalSummary(round=round_no, duration_ms=ms, **fields).model_dump(
                mode="json"
            )

        try:
            args = parse_policy_search_args(call.get("args"))
        except PolicySearchArgumentError as exc:
            message = ToolMessage(
                content=invalid_arguments_text(exc.reason),
                tool_call_id=call["id"],
                name=SEARCH_POLICY_KNOWLEDGE,
                status="error",
                artifact={ARTIFACT_KEY: []},
            )
            retrievals.append(
                summary(
                    as_of=None,
                    result_count=0,
                    citations=[],
                    outcome="invalid_arguments",
                    error_code="invalid_arguments",
                    rejected_argument_names=exc.rejected,
                )
            )
            return {
                "messages": [ai, message],
                "pending": None,
                "pending_kind": None,
                "retrievals": retrievals,
                "policy_retrieval_status": merge_status(status, "invalid"),
            }

        try:
            result = self._run_retriever(args.query, context, args.as_of)
        except _RetrievalCheckError as exc:
            detail = exc.detail
        except EmbeddingError as exc:
            detail = exc.code
        except SQLAlchemyError:
            detail = "database_unavailable"
        except Exception as exc:  # noqa: BLE001 - fail closed; never "no policy"
            logger.warning("policy retrieval failed", extra={"error_type": type(exc).__name__})
            detail = "retrieval_failed"
        else:
            detail = None
        if detail is not None:
            retrievals.append(
                summary(
                    as_of=None, result_count=0, citations=[], outcome="error", error_code=detail
                )
            )
            # Terminal: nothing appended, the model is not called again.
            return {
                "pending": None,
                "pending_kind": None,
                "retrievals": retrievals,
                "policy_retrieval_status": "error",
                "error": _error("agent_retrieval_error", detail),
            }

        citations = [r.citation for r in result.results]
        message = ToolMessage(
            content=format_policy_results(result),
            tool_call_id=call["id"],
            name=SEARCH_POLICY_KNOWLEDGE,
            status="success",
            artifact={ARTIFACT_KEY: citations},
        )
        sources = list(state.get("policy_sources", []))
        known = {s["citation"] for s in sources}
        for r in result.results:
            if r.citation not in known:
                known.add(r.citation)
                sources.append(
                    {
                        "citation": r.citation,
                        "title": r.title,
                        "document_key": r.document_key,
                        "version": r.version,
                        "section": r.section,
                        "effective_from": r.effective_from.isoformat(),
                        "effective_to": r.effective_to.isoformat() if r.effective_to else None,
                    }
                )
        outcome = "success" if citations else "no_results"
        retrievals.append(
            summary(
                as_of=result.as_of,
                result_count=len(citations),
                citations=citations,
                outcome=outcome,
            )
        )
        # One update: the original AIMessage and its retrieval ToolMessage.
        return {
            "messages": [ai, message],
            "pending": None,
            "pending_kind": None,
            "retrievals": retrievals,
            "policy_sources": sources,
            "policy_retrieval_status": merge_status(status, outcome),
        }

    def _run_retriever(
        self, query: str, context: AgentContext, as_of: date | None
    ) -> RetrievalResult:
        if self._retriever is None:
            raise _RetrievalCheckError("retriever_not_configured")
        result = self._retriever().retrieve(
            query, context, as_of=as_of, limit=POLICY_RETRIEVAL_LIMIT
        )
        if not isinstance(result, RetrievalResult):
            raise _RetrievalCheckError("invalid_retrieval_result")
        if as_of is not None and result.as_of != as_of:
            raise _RetrievalCheckError("ineligible_version")
        if len(result.results) > POLICY_RETRIEVAL_LIMIT:
            raise _RetrievalCheckError("limit_exceeded")
        seen: set[str] = set()
        for r in result.results:
            # Defence in depth behind the SQL filter: every chunk must be effective on as_of.
            if not (
                r.effective_from <= result.as_of
                and (r.effective_to is None or result.as_of < r.effective_to)
            ):
                raise _RetrievalCheckError("ineligible_version")
            if r.citation in seen:
                raise _RetrievalCheckError("duplicate_citation")
            seen.add(r.citation)
        return result
