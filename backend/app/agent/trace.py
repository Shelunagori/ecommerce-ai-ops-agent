"""Agent Execution Trace: the ordered, SAFE record of what one assistant run actually did.

Built by the runner from the graph's own execution records (checkpointed state): the model
round counter, the commerce tool / policy retrieval / action proposal summaries each tagged
with the model round that requested them, the grounding result, and the approval/execution
state of the action. It is never inferred from the answer text.

Ordering: every model call is one round; the capability batch a round requested runs before
the next round, so ``model(r) -> capabilities(r) -> model(r+1) ...`` is the executed order.

The per-step builders below (``request_step``, ``tool_step``, ...) are shared with the LIVE
run events (``app.agent.events``): a streamed step and the final trace entry for it are built
by the same function, so the live timeline and the final trace cannot disagree.

Safety: labels are fixed strings and metadata is a closed set of identifiers, counts,
outcomes and measured durations. Never included: prompts, messages, model output or reasoning,
tool arguments, retrieved text, SQL, embeddings, tenant ids, tokens or connection strings.
Durations are included only where the system measured them (tools, retrievals, total run).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TraceEventKind(StrEnum):
    REQUEST = "request"
    MODEL = "model"
    COMMERCE_TOOL = "commerce_tool"
    RETRIEVAL = "retrieval"
    GROUNDING = "grounding"
    ACTION_PROPOSAL = "action_proposal"
    APPROVAL = "approval"
    ACTION_EXECUTION = "action_execution"
    CHECKPOINT = "checkpoint"
    RESPONSE = "response"


class TraceStatus(StrEnum):
    COMPLETED = "completed"
    WAITING = "waiting"  # e.g. human approval required
    REJECTED = "rejected"  # refused by a rule or by the human
    FAILED = "failed"


MetaValue = str | int | float | bool | None


class ExecutionTraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=1)
    kind: TraceEventKind
    label: str = Field(max_length=80)
    status: TraceStatus
    detail: str | None = Field(default=None, max_length=160)
    metadata: dict[str, MetaValue] = {}


@dataclass(frozen=True)
class TraceStep:
    """One finished step (no sequence yet): the unit shared by the final trace and the live
    run events. ``metadata`` drops ``None`` values."""

    kind: TraceEventKind
    label: str
    status: TraceStatus = TraceStatus.COMPLETED
    detail: str | None = None
    metadata: dict[str, MetaValue] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        kind: TraceEventKind,
        label: str,
        status: TraceStatus = TraceStatus.COMPLETED,
        detail: str | None = None,
        **metadata: MetaValue,
    ) -> TraceStep:
        return cls(
            kind, label, status, detail, {k: v for k, v in metadata.items() if v is not None}
        )


_ACTION_TYPES = {
    "propose_cancel_order": "cancel_order",
    "propose_store_credit": "issue_store_credit",
}
TOOL_FAILURES = frozenset(
    {"service_unavailable", "internal_error", "unknown_tool", "invalid_arguments"}
)
_RETRIEVERS = {
    "semantic-pgvector-v1": "semantic (pgvector)",
    "lexical-pg-fts-v1": "lexical (PostgreSQL full-text)",
}
REQUEST_DETAIL = "Tenant scope from the verified principal"
PUBLIC_DEMO_REQUEST_DETAIL = f"{REQUEST_DETAIL} · read-only public demo (no action tools)"
MODEL_LABEL = "Agent orchestration"
RETRIEVAL_LABEL = "Policy retrieval"
GROUNDING_LABEL = "Grounding validation"
PROPOSAL_LABEL = "Action proposal"
EXECUTION_LABEL = "Deterministic execution"
RESPONSE_LABEL = "Response generated"


# --- per-step builders (shared with app.agent.events) -----------------------------------------
def request_step(*, public_demo: bool = False) -> TraceStep:
    detail = PUBLIC_DEMO_REQUEST_DETAIL if public_demo else REQUEST_DETAIL
    return TraceStep.of(TraceEventKind.REQUEST, "Request received", detail=detail)


def model_detail(tool_names: Iterable[str], *, retrieval: bool, proposal: bool, final: bool) -> str:
    names = ", ".join(dict.fromkeys(tool_names))
    if names:
        return f"Requested commerce tool: {names}"
    if retrieval:
        return "Requested policy retrieval"
    if proposal:
        return "Proposed an approval-gated action"
    return "Composed the final answer" if final else "Model turn"


def model_step(round_no: int, provider: str, detail: str) -> TraceStep:
    return TraceStep.of(
        TraceEventKind.MODEL, MODEL_LABEL, detail=detail, call=round_no, provider=provider
    )


def tool_step(summary: dict[str, Any]) -> TraceStep:
    outcome = str(summary.get("outcome"))
    failed = outcome in TOOL_FAILURES
    return TraceStep.of(
        TraceEventKind.COMMERCE_TOOL,
        f"Tool: {summary.get('tool')}",
        TraceStatus.FAILED if failed else TraceStatus.COMPLETED,
        detail=(
            f"Tool call refused or failed ({outcome})"
            if failed
            else "Deterministic, tenant-scoped PostgreSQL lookup"
        ),
        tool=str(summary.get("tool")),
        outcome=outcome,
        duration_ms=summary.get("duration_ms"),
    )


def retrieval_step(summary: dict[str, Any]) -> TraceStep:
    outcome = str(summary.get("outcome"))
    count = int(summary.get("result_count") or 0)
    detail = {
        "success": f"{count} eligible policy section{'s' if count != 1 else ''}",
        "no_results": "No eligible policy section for the date",
        "invalid_arguments": "Retrieval request refused (invalid arguments)",
    }.get(outcome, "Policy retrieval service was unavailable")
    return TraceStep.of(
        TraceEventKind.RETRIEVAL,
        RETRIEVAL_LABEL,
        TraceStatus.COMPLETED if outcome in ("success", "no_results") else TraceStatus.FAILED,
        detail=detail,
        outcome=outcome,
        result_count=count,
        retriever=summary.get("retriever"),
        retrieval_mode=_RETRIEVERS.get(str(summary.get("retriever"))),
        as_of=summary.get("as_of"),
        duration_ms=summary.get("duration_ms"),
    )


def proposal_step(summary: dict[str, Any]) -> TraceStep:
    outcome = summary.get("outcome")
    if outcome == "error":
        return TraceStep.of(
            TraceEventKind.ACTION_PROPOSAL,
            PROPOSAL_LABEL,
            TraceStatus.FAILED,
            detail="The action service was unavailable; nothing was persisted",
            action_type=_ACTION_TYPES.get(str(summary.get("capability"))),
            outcome="error",
        )
    refused = outcome != "pending_approval"
    return TraceStep.of(
        TraceEventKind.ACTION_PROPOSAL,
        PROPOSAL_LABEL,
        TraceStatus.REJECTED if refused else TraceStatus.COMPLETED,
        detail=(
            f"Refused by server-side rules ({summary.get('error_code')})"
            if refused
            else "Validated and persisted as a pending request"
        ),
        action_type=_ACTION_TYPES.get(str(summary.get("capability"))),
        outcome=outcome,
        error_code=summary.get("error_code") if refused else None,
    )


def grounding_step(cited: int) -> TraceStep:
    return TraceStep.of(
        TraceEventKind.GROUNDING,
        GROUNDING_LABEL,
        detail=f"{cited} citation{'s' if cited != 1 else ''} checked against this run",
        citations_verified=cited,
    )


def approval_waiting_step(action: dict[str, Any]) -> TraceStep:
    return TraceStep.of(
        TraceEventKind.APPROVAL,
        "Human approval required",
        TraceStatus.WAITING,
        detail="Nothing changes until an approver decides",
        action_type=action.get("action_type"),
        action_status=action.get("status"),
    )


def checkpoint_step(durable: bool) -> TraceStep:
    return TraceStep.of(
        TraceEventKind.CHECKPOINT,
        "Run paused · state checkpointed",
        detail="Resumable after a restart" if durable else "In-memory",
        durable=durable,
    )


def paused_response_step(model_calls: int, duration_ms: float) -> TraceStep:
    return TraceStep.of(
        TraceEventKind.RESPONSE,
        "Approval request returned",
        TraceStatus.WAITING,
        model_calls=model_calls,
        duration_ms=duration_ms,
    )


def decision_step(action: dict[str, Any]) -> TraceStep:
    """The recorded human decision. ``decision`` (approved / rejected / expired) is stable
    whether it is read right after the decision (live) or after execution (final trace)."""
    status = str(action.get("status"))
    decision = status if status in ("rejected", "expired") else "approved"
    label, trace_status, detail = {
        "rejected": ("Rejected by a human", TraceStatus.REJECTED, "Nothing was changed"),
        "expired": ("Approval expired", TraceStatus.REJECTED, "Nothing was changed"),
    }.get(status, ("Approved by a human", TraceStatus.COMPLETED, None))
    return TraceStep.of(
        TraceEventKind.APPROVAL,
        label,
        trace_status,
        detail=detail,
        action_type=action.get("action_type"),
        decision=decision,
    )


EXECUTED_STATUSES = frozenset({"succeeded", "failed", "executing", "approved"})


def execution_step(action: dict[str, Any]) -> TraceStep:
    status = str(action.get("status"))
    ok = status == "succeeded"
    return TraceStep.of(
        TraceEventKind.ACTION_EXECUTION,
        EXECUTION_LABEL,
        TraceStatus.COMPLETED if ok else TraceStatus.FAILED,
        detail=(
            "Executed once by application code; audit event recorded"
            if ok
            else f"Not completed ({action.get('failure_code') or status})"
        ),
        action_type=action.get("action_type"),
        action_status=status,
        failure_code=action.get("failure_code"),
        audit_recorded=True if ok else None,
    )


def response_step(
    *, outcome: bool, model_calls: int, tool_calls: int, citations: int, duration_ms: float
) -> TraceStep:
    return TraceStep.of(
        TraceEventKind.RESPONSE,
        "Outcome recorded" if outcome else RESPONSE_LABEL,
        model_calls=model_calls,
        tool_calls=tool_calls,
        citations=citations,
        duration_ms=duration_ms,
    )


def grounding_ran(values: dict[str, Any], *, paused: bool, action: dict[str, Any] | None) -> bool:
    """Grounding runs on an accepted model ANSWER after policy retrieval; approval outcomes are
    templated, not model answers, so they have no grounding step."""
    answered = not paused and action is None
    return answered and values.get("policy_retrieval_status") in ("success", "no_results")


# --- the final trace ---------------------------------------------------------------------------
class _Trace:
    def __init__(self) -> None:
        self.events: list[ExecutionTraceEvent] = []

    def add(self, step: TraceStep) -> None:
        self.events.append(
            ExecutionTraceEvent(
                sequence=len(self.events) + 1,
                kind=step.kind,
                label=step.label,
                status=step.status,
                detail=step.detail,
                metadata=dict(step.metadata),
            )
        )


def build_execution_trace(
    values: dict[str, Any],
    *,
    provider: str,
    duration_ms: float,
    checkpointed: bool,
    durable_checkpoints: bool,
    paused: bool,
    action: dict[str, Any] | None,
    public_demo: bool = False,
) -> list[ExecutionTraceEvent]:
    """``values``: the graph state after the run (or after resuming it)."""
    trace = _Trace()
    tool_calls = values.get("tool_calls", [])
    retrievals = values.get("retrievals", [])
    proposals = values.get("action_calls", [])
    model_calls = int(values.get("model_calls", 0))

    trace.add(request_step(public_demo=public_demo))
    for round_no in range(1, model_calls + 1):
        r_tools = [t for t in tool_calls if t.get("round") == round_no]
        r_retrievals = [r for r in retrievals if r.get("round") == round_no]
        r_proposals = [p for p in proposals if p.get("round") == round_no]
        final = round_no == model_calls and not (r_tools or r_retrievals or r_proposals)
        detail = model_detail(
            (str(t.get("tool")) for t in r_tools),
            retrieval=bool(r_retrievals),
            proposal=bool(r_proposals),
            final=final,
        )
        trace.add(model_step(round_no, provider, detail))
        for t in r_tools:
            trace.add(tool_step(t))
        for r in r_retrievals:
            trace.add(retrieval_step(r))
        for p in r_proposals:
            trace.add(proposal_step(p))

    if grounding_ran(values, paused=paused, action=action):
        trace.add(grounding_step(len(values.get("citations", []))))

    if paused and action is not None:
        trace.add(approval_waiting_step(action))
        if checkpointed:
            trace.add(checkpoint_step(durable_checkpoints))
        trace.add(paused_response_step(model_calls, duration_ms))
        return trace.events

    if action is not None:  # resumed after a human decision
        trace.add(decision_step(action))
        if str(action.get("status")) in EXECUTED_STATUSES:
            trace.add(execution_step(action))

    trace.add(
        response_step(
            outcome=action is not None,
            model_calls=model_calls,
            tool_calls=len(tool_calls),
            citations=len(values.get("citations", [])),
            duration_ms=duration_ms,
        )
    )
    return trace.events
