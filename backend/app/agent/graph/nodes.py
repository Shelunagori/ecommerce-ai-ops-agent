"""MODEL, TOOLS and RETRIEVE nodes. Each returns only a state update.

External side effects: MODEL invokes the Step-4 provider; TOOLS executes the read-only
Step-3 tools through the hardened ``ToolExecutor`` (registry allowlist, trusted runtime
injection, argument checks, sanitised summaries, call-id checks, Step-3 logging);
RETRIEVE (Step 9) runs the policy retriever with the trusted tenant context and a fixed
limit. Policy retrieval never goes through ``ToolExecutor``.
"""

import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import date
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from sqlalchemy.exc import SQLAlchemyError

from app.actions import errors as action_errors
from app.actions.capability import ACTION_TOOLS, action_tools, parse_action_call
from app.agent.assistant.executor import ToolExecutionError, ToolExecutor, safe_name
from app.agent.assistant.limits import AssistantLimits
from app.agent.assistant.result import RetrievalSummary
from app.agent.context import AgentContext
from app.agent.events import RunEventType, current_emitter
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
from app.agent.rag.grounding import check_citations, check_grounding
from app.agent.trace import (
    EXECUTED_STATUSES,
    EXECUTION_LABEL,
    GROUNDING_LABEL,
    MODEL_LABEL,
    PROPOSAL_LABEL,
    RETRIEVAL_LABEL,
    TraceEventKind,
    decision_step,
    execution_step,
    grounding_step,
    model_detail,
    model_step,
    proposal_step,
    retrieval_step,
    tool_step,
)
from app.knowledge.embeddings.errors import EmbeddingError
from app.knowledge.retrieval import RetrievalResult

logger = logging.getLogger("app.agent.graph")

THREAD_SCOPE_MESSAGE = "This conversation thread belongs to a different account."
ARTIFACT_KEY = "policy_citations"  # ToolMessage.artifact: kept in history, never sent to models
ACTION_NAMES = frozenset(ACTION_TOOLS)
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
        actions: Callable[[], Any] | None = None,
    ) -> None:
        self._provider = provider
        self._executor = executor
        self._limits = limits
        self._profile = profile
        self._retriever = retriever
        self._actions = actions  # zero-arg factory -> ActionService (Step 10)
        self._bound: tuple[BaseTool, ...] = (
            executor.tools
            + ((policy_search_tool(),) if profile.policy_knowledge else ())
            + (action_tools() if profile.actions else ())
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
        events = current_emitter()
        provider = self._provider.info.provider
        step = events.start(
            TraceEventKind.MODEL,
            MODEL_LABEL,
            "Deciding the next step",
            call=round_no,
            provider=provider,
        )
        try:
            chat = self._provider.invoke_chat(
                state["messages"],
                tools=self._bound,
                operation=prompt.PROMPT_ID,
                prompt_version=prompt.PROMPT_VERSION,
            )
        except LLMError as exc:
            events.fail(
                step,
                TraceEventKind.MODEL,
                MODEL_LABEL,
                f"Model provider call failed ({exc.code})",
                call=round_no,
                provider=provider,
            )
            return {**update, "pending": None, "error": _error(exc.code, "llm", exc.message)}
        ai: AIMessage = chat.message  # the provider's original message, never rebuilt

        status = state.get("policy_retrieval_status", "none")
        decision = evaluate_model_turn(
            ai,
            round_no=round_no,
            seen_call_ids=state.get("seen_tool_call_ids", []),
            # every capability call counts: commerce + policy retrieval (incl. invalid ones)
            tool_calls_so_far=len(state.get("tool_calls", []))
            + len(state.get("retrievals", []))
            + len(state.get("action_calls", [])),
            limits=self._limits,
            retrieval_name=SEARCH_POLICY_KNOWLEDGE if self._profile.policy_knowledge else None,
            retrieval_done=status in ("success", "no_results"),
            action_names=ACTION_NAMES if self._profile.actions else frozenset(),
            action_attempts=len(state.get("action_calls", [])),
        )
        if decision.kind in ("tools", "retrieve", "action"):
            names = (
                [safe_name(c.get("name")) or "<invalid>" for c in ai.tool_calls]
                if decision.kind == "tools"
                else []
            )
            detail = model_detail(
                names,
                retrieval=decision.kind == "retrieve",
                proposal=decision.kind == "action",
                final=False,
            )
            events.finish(step, model_step(round_no, provider, detail))
            seen = [*state.get("seen_tool_call_ids", []), *decision.new_call_ids]
            kind = {"retrieve": "retrieval", "action": "action"}.get(decision.kind, "commerce")
            return {**update, "pending": ai, "pending_kind": kind, "seen_tool_call_ids": seen}
        if decision.kind == "answer":
            final_detail = model_detail([], retrieval=False, proposal=False, final=True)
            events.finish(step, model_step(round_no, provider, final_detail))
            if self._profile.policy_knowledge:
                sources = {s["citation"]: s for s in state.get("policy_sources", [])}
                # Reported only where the final trace reports it: after policy retrieval (or
                # when validation actually rejects the answer).
                reported = status in ("success", "no_results")
                g_step = (
                    events.start(
                        TraceEventKind.GROUNDING,
                        GROUNDING_LABEL,
                        "Checking citations against this run's sources",
                    )
                    if reported
                    else None
                )
                grounding = check_grounding(
                    decision.answer or "",
                    status=status,  # type: ignore[arg-type]
                    current=sources,
                    earlier=earlier_policy_citations(state["messages"]),
                )
                if not grounding.ok:
                    events.fail(
                        g_step,
                        TraceEventKind.GROUNDING,
                        GROUNDING_LABEL,
                        "The answer failed citation validation and was not returned",
                    )
                    # The ungrounded answer is neither returned nor kept in history.
                    return {
                        **update,
                        "pending": None,
                        "error": _error("agent_grounding_error", grounding.detail),
                    }
                if reported:
                    events.finish(g_step, grounding_step(len(grounding.citations)))
                update["citations"] = grounding.citations
            return {**update, "pending": None, "messages": [ai], "answer": decision.answer}
        events.fail(
            step,
            TraceEventKind.MODEL,
            MODEL_LABEL,
            f"Model response rejected ({decision.error_code or 'agent_protocol_error'})",
            call=round_no,
            provider=provider,
        )
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
        events = current_emitter()
        for call in ai.tool_calls:  # sequential, in request order; no model call in between
            name = safe_name(call.get("name")) or "<invalid>"
            step = events.start(
                TraceEventKind.COMMERCE_TOOL,
                f"Tool: {name}",
                "Deterministic, tenant-scoped PostgreSQL lookup",
                tool=name,
            )
            try:
                message, summary = self._executor.execute(call, context, round_no)
            except ToolExecutionError:
                events.fail(
                    step,
                    TraceEventKind.COMMERCE_TOOL,
                    f"Tool: {name}",
                    "The tool result could not be processed",
                    tool=name,
                )
                return {
                    "pending": None,
                    "pending_kind": None,
                    "tool_calls": summaries,
                    "error": _error("agent_protocol_error", "tool_result"),
                }
            tool_messages.append(message)
            summaries.append(summary.model_dump(mode="json"))
            events.finish(step, tool_step(summaries[-1]), duration_ms=summary.duration_ms)
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
        events = current_emitter()
        step = events.start(
            TraceEventKind.RETRIEVAL, RETRIEVAL_LABEL, "Searching tenant-scoped policy knowledge"
        )

        def report(entry: dict[str, Any]) -> None:
            events.finish(step, retrieval_step(entry), duration_ms=entry.get("duration_ms"))

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
            report(retrievals[-1])
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
            report(retrievals[-1])
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
                retriever=result.retriever,
            )
        )
        report(retrievals[-1])
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


# --- Step 10: approval-gated actions ------------------------------------------------------------
def _json_tool_message(
    call_id: str, name: str, payload: dict[str, Any], *, error: bool
) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=call_id,
        name=name,
        status="error" if error else "success",
    )


def outcome_text(view: dict[str, Any]) -> str:
    """Application-written outcome message (never model-written: a write can't be misreported)."""
    ref = f"(action {view['id']})"
    status = view["status"]
    result = view.get("result") or {}
    if status == "succeeded" and view["action_type"] == "cancel_order":
        return (
            f"Done: order {result['order_number']} was cancelled "
            f"(previous status: {result['previous_status']}) {ref}."
        )
    if status == "succeeded":
        target = f" for order {result['order_number']}" if result.get("order_number") else ""
        evidence = " ".join(f"[{e['citation']}]" for e in view.get("evidence", []))
        return (
            f"Done: store credit of {result['amount']} {result['currency']} was issued to "
            f"customer {result['customer_code']}{target} {ref}. Policy basis: {evidence}"
        ).strip()
    if status == "rejected":
        return f"The request was rejected; nothing was changed {ref}."
    if status == "expired":
        return f"The approval window expired; nothing was changed {ref}."
    code = view.get("failure_code") or "execution_failed"
    return f"The action could not be executed ({code}); nothing was changed {ref}."


def approval_text(view: dict[str, Any]) -> str:
    return (
        "This action needs your approval before anything is changed: "
        f"{view['summary']} (action {view['id']}, expires {view['expires_at']})."
    )


class ActionNodes:
    """PROPOSE -> APPROVAL (interrupt) -> EXECUTE. Business writes happen ONLY in EXECUTE,
    only through ``ActionService.execute``, and only for a request a human approved."""

    def __init__(self, nodes: GraphNodes) -> None:
        self._n = nodes

    def _service(self) -> Any:
        if self._n._actions is None:
            raise RuntimeError("action service not configured")
        return self._n._actions()

    def propose(
        self, state: CommerceGraphState, config: RunnableConfig, runtime: Runtime[AgentContext]
    ) -> dict[str, Any]:
        context = _trusted_context(runtime)
        ai = state.get("pending")
        if ai is None or state.get("pending_kind") != "action" or len(ai.tool_calls) != 1:
            raise RuntimeError("propose node reached without one approved proposal call")
        call = ai.tool_calls[0]
        round_no = state.get("model_calls", 0)
        attempts = list(state.get("action_calls", []))
        summary: dict[str, Any] = {"round": round_no, "capability": call.get("name")}
        events = current_emitter()
        step = events.start(
            TraceEventKind.ACTION_PROPOSAL, PROPOSAL_LABEL, "Validating against server-side rules"
        )
        try:
            proposal = parse_action_call(call.get("name", ""), call.get("args"))
            sources = {s["citation"]: s for s in state.get("policy_sources", [])}
            problem = check_citations(
                proposal.citations,
                current=sources,
                earlier=earlier_policy_citations(state["messages"]),
            )
            if problem is not None:
                raise action_errors.ActionArgumentsInvalidError(
                    "policy_citations must come from policy retrieved in this request.",
                    detail=problem,
                )
            provider = self._n._provider.info
            view = self._service().propose(
                context.tenant,
                proposal.action_type,
                proposal.arguments,
                evidence=[sources[c] for c in proposal.citations],
                requested_by={
                    "runner": "langgraph",
                    "provider": provider.provider,
                    "model": provider.model,
                    "prompt_version": self._n._profile.prompt_version,
                },
                thread_key=(config.get("configurable") or {}).get("thread_id"),
                tool_call_id=call["id"],
            )
        except action_errors.ActionError as exc:
            if exc.status_code >= 500:
                raise
            # Invalid / not allowed: nothing persisted; the model may explain or correct.
            code = exc.detail if exc.code == "action_invalid_arguments" and exc.detail else exc.code
            attempts.append({**summary, "outcome": "rejected", "error_code": code})
            events.finish(step, proposal_step(attempts[-1]))
            message = _json_tool_message(
                call["id"],
                call.get("name") or "action",
                {"ok": False, "error": {"code": code, "message": exc.message}},
                error=True,
            )
            return {
                "messages": [ai, message],
                "pending": None,
                "pending_kind": None,
                "action_calls": attempts,
            }
        except SQLAlchemyError:
            attempts.append({**summary, "outcome": "error", "error_code": "database_unavailable"})
            events.finish(step, proposal_step(attempts[-1]))
            return {
                "pending": None,
                "pending_kind": None,
                "action_calls": attempts,
                "error": _error("agent_action_error", "database_unavailable"),
            }
        attempts.append({**summary, "outcome": "pending_approval", "action_id": str(view.id)})
        events.finish(step, proposal_step(attempts[-1]))
        # The proposal AIMessage stays in ``pending`` (checkpointed) until EXECUTE appends it
        # together with its ToolMessage, so history is never left with an unanswered call.
        return {"pending_action": view.as_dict(), "action_calls": attempts}

    def await_approval(self, state: CommerceGraphState) -> dict[str, Any]:
        """Pure: pause for a human. Re-runs on resume (LangGraph semantics), so it must
        have no side effects. The decision itself is recorded in the DB by the runner."""
        pending = state.get("pending_action")
        if pending is None:
            raise RuntimeError("approval node reached without a pending action")
        signal = interrupt({"type": "action_approval", "action": pending})
        if not isinstance(signal, dict) or signal.get("action_request_id") != pending["id"]:
            return {"error": _error("agent_protocol_error", "approval_mismatch"), "pending": None}
        return {}

    def execute(self, state: CommerceGraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        context = _trusted_context(runtime)
        if state.get("error") is not None:
            return {}
        ai = state.get("pending")
        pending = state.get("pending_action")
        if ai is None or pending is None:
            raise RuntimeError("execute node reached without a pending action")
        service = self._service()
        action_id = uuid.UUID(pending["id"])
        events = current_emitter()
        step: str | None = None
        executing = False
        try:
            view = service.get(context.tenant, action_id)
            if view.status != "pending_approval":
                # The human decision as recorded in PostgreSQL (the authority), not the UI's.
                events.finish(
                    None,
                    decision_step(view.as_dict()),
                    event_type=RunEventType.APPROVAL_RESOLVED,
                )
            if view.status in ("approved", "executing", "succeeded"):
                executing = True
                step = events.start(
                    TraceEventKind.ACTION_EXECUTION,
                    EXECUTION_LABEL,
                    "Executing once in one database transaction",
                    action_type=view.action_type,
                )
                try:
                    view = service.execute(context.tenant, action_id)
                except action_errors.ActionError:
                    # failed / expired / precondition changed: report the DB's truth
                    view = service.get(context.tenant, action_id)
                if view.status == "executing":  # someone else is executing it right now
                    events.fail(
                        step,
                        TraceEventKind.ACTION_EXECUTION,
                        EXECUTION_LABEL,
                        "Another request is executing this action right now",
                    )
                    return {
                        "pending": None,
                        "error": _error("agent_action_error", "action_in_progress"),
                    }
            elif view.status == "pending_approval":
                return {
                    "pending": None,
                    "error": _error("agent_protocol_error", "approval_not_decided"),
                }
        except action_errors.ActionError as exc:
            if executing:
                events.fail(
                    step,
                    TraceEventKind.ACTION_EXECUTION,
                    EXECUTION_LABEL,
                    f"The action service refused the request ({exc.code})",
                )
            return {"pending": None, "error": _error("agent_action_error", exc.code)}
        except SQLAlchemyError:
            if executing:
                events.fail(
                    step,
                    TraceEventKind.ACTION_EXECUTION,
                    EXECUTION_LABEL,
                    "The action service was unavailable; the approved request stays retryable",
                )
            return {"pending": None, "error": _error("agent_action_error", "database_unavailable")}
        final = view.as_dict()
        if executing and final["status"] in EXECUTED_STATUSES:
            events.finish(step, execution_step(final))
        elif final["status"] not in EXECUTED_STATUSES:
            events.skip(
                TraceEventKind.ACTION_EXECUTION,
                EXECUTION_LABEL,
                "Not executed: nothing was changed",
            )
        call = ai.tool_calls[0]
        tool_message = _json_tool_message(
            call["id"],
            call.get("name") or "action",
            {
                "ok": final["status"] == "succeeded",
                "action": {
                    "id": final["id"],
                    "status": final["status"],
                    "result": final["result"],
                    "failure_code": final["failure_code"],
                },
            },
            error=final["status"] != "succeeded",
        )
        text = outcome_text(final)
        return {
            "messages": [
                ai,
                tool_message,
                AIMessage(content=text, id=f"action-outcome-{final['id']}"),
            ],
            "pending": None,
            "pending_kind": None,
            "pending_action": None,
            "action": final,
            "answer": text,
            "citations": [e["citation"] for e in final.get("evidence", [])]
            if final["status"] == "succeeded"
            else [],
        }
