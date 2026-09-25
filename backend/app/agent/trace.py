"""Agent Execution Trace: the ordered, SAFE record of what one assistant run actually did.

Built by the runner from the graph's own execution records (checkpointed state): the model
round counter, the commerce tool / policy retrieval / action proposal summaries each tagged
with the model round that requested them, the grounding result, and the approval/execution
state of the action. It is never inferred from the answer text.

Ordering: every model call is one round; the capability batch a round requested runs before
the next round, so ``model(r) -> capabilities(r) -> model(r+1) ...`` is the executed order.

Safety: labels are fixed strings and metadata is a closed set of identifiers, counts,
outcomes and measured durations. Never included: prompts, messages, model output or reasoning,
tool arguments, retrieved text, SQL, embeddings, tenant ids, tokens or connection strings.
Durations are included only where the system measured them (tools, retrievals, total run).
"""

from __future__ import annotations

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


_ACTION_TYPES = {
    "propose_cancel_order": "cancel_order",
    "propose_store_credit": "issue_store_credit",
}
_TOOL_FAILURES = {"service_unavailable", "internal_error", "unknown_tool", "invalid_arguments"}
_RETRIEVERS = {
    "semantic-pgvector-v1": "semantic (pgvector)",
    "lexical-pg-fts-v1": "lexical (PostgreSQL full-text)",
}


class _Trace:
    def __init__(self) -> None:
        self.events: list[ExecutionTraceEvent] = []

    def add(
        self,
        kind: TraceEventKind,
        label: str,
        status: TraceStatus = TraceStatus.COMPLETED,
        detail: str | None = None,
        **metadata: MetaValue,
    ) -> None:
        self.events.append(
            ExecutionTraceEvent(
                sequence=len(self.events) + 1,
                kind=kind,
                label=label,
                status=status,
                detail=detail,
                metadata={k: v for k, v in metadata.items() if v is not None},
            )
        )


def _model_detail(tools: list, retrievals: list, proposals: list, final: bool) -> str:
    if tools:
        names = ", ".join(dict.fromkeys(str(t.get("tool")) for t in tools))
        return f"Requested commerce tool: {names}"
    if retrievals:
        return "Requested policy retrieval"
    if proposals:
        return "Proposed an approval-gated action"
    return "Composed the final answer" if final else "Model turn"


def build_execution_trace(
    values: dict[str, Any],
    *,
    provider: str,
    duration_ms: float,
    checkpointed: bool,
    durable_checkpoints: bool,
    paused: bool,
    action: dict[str, Any] | None,
) -> list[ExecutionTraceEvent]:
    """``values``: the graph state after the run (or after resuming it)."""
    trace = _Trace()
    tool_calls = values.get("tool_calls", [])
    retrievals = values.get("retrievals", [])
    proposals = values.get("action_calls", [])
    model_calls = int(values.get("model_calls", 0))

    trace.add(
        TraceEventKind.REQUEST,
        "Request received",
        detail="Tenant scope from the verified principal",
    )
    for round_no in range(1, model_calls + 1):
        r_tools = [t for t in tool_calls if t.get("round") == round_no]
        r_retrievals = [r for r in retrievals if r.get("round") == round_no]
        r_proposals = [p for p in proposals if p.get("round") == round_no]
        final = round_no == model_calls and not (r_tools or r_retrievals or r_proposals)
        trace.add(
            TraceEventKind.MODEL,
            "Agent orchestration",
            detail=_model_detail(r_tools, r_retrievals, r_proposals, final),
            call=round_no,
            provider=provider,
        )
        for t in r_tools:
            outcome = str(t.get("outcome"))
            failed = outcome in _TOOL_FAILURES
            trace.add(
                TraceEventKind.COMMERCE_TOOL,
                f"Tool: {t.get('tool')}",
                TraceStatus.FAILED if failed else TraceStatus.COMPLETED,
                detail=(
                    f"Tool call refused or failed ({outcome})"
                    if failed
                    else "Deterministic, tenant-scoped PostgreSQL lookup"
                ),
                tool=str(t.get("tool")),
                outcome=outcome,
                duration_ms=t.get("duration_ms"),
            )
        for r in r_retrievals:
            outcome = str(r.get("outcome"))
            count = int(r.get("result_count") or 0)
            detail = {
                "success": f"{count} eligible policy section{'s' if count != 1 else ''}",
                "no_results": "No eligible policy section for the date",
                "invalid_arguments": "Retrieval request refused (invalid arguments)",
            }.get(outcome, "Retrieval failed")
            trace.add(
                TraceEventKind.RETRIEVAL,
                "Policy retrieval",
                TraceStatus.COMPLETED
                if outcome in ("success", "no_results")
                else TraceStatus.FAILED,
                detail=detail,
                outcome=outcome,
                result_count=count,
                retriever=r.get("retriever"),
                retrieval_mode=_RETRIEVERS.get(str(r.get("retriever"))),
                as_of=r.get("as_of"),
                duration_ms=r.get("duration_ms"),
            )
        for p in r_proposals:
            refused = p.get("outcome") != "pending_approval"
            trace.add(
                TraceEventKind.ACTION_PROPOSAL,
                "Action proposal",
                TraceStatus.REJECTED if refused else TraceStatus.COMPLETED,
                detail=(
                    f"Refused by server-side rules ({p.get('error_code')})"
                    if refused
                    else "Validated and persisted as a pending request"
                ),
                action_type=_ACTION_TYPES.get(str(p.get("capability"))),
                outcome=p.get("outcome"),
                error_code=p.get("error_code") if refused else None,
            )

    # Grounding runs on an accepted model ANSWER; approval outcomes are templated, not model
    # answers, so they have no grounding step.
    answered = not paused and action is None
    if answered and values.get("policy_retrieval_status") in ("success", "no_results"):
        cited = len(values.get("citations", []))
        trace.add(
            TraceEventKind.GROUNDING,
            "Grounding validation",
            detail=f"{cited} citation{'s' if cited != 1 else ''} checked against this run",
            citations_verified=cited,
        )

    if paused and action is not None:
        trace.add(
            TraceEventKind.APPROVAL,
            "Human approval required",
            TraceStatus.WAITING,
            detail="Nothing changes until an approver decides",
            action_type=action.get("action_type"),
            action_status=action.get("status"),
        )
        if checkpointed:
            trace.add(
                TraceEventKind.CHECKPOINT,
                "Run paused · state checkpointed",
                detail="Resumable after a restart" if durable_checkpoints else "In-memory",
                durable=durable_checkpoints,
            )
        trace.add(
            TraceEventKind.RESPONSE,
            "Approval request returned",
            TraceStatus.WAITING,
            model_calls=model_calls,
            duration_ms=duration_ms,
        )
        return trace.events

    if action is not None:  # resumed after a human decision
        status = str(action.get("status"))
        decided = {
            "rejected": ("Rejected by a human", TraceStatus.REJECTED, "Nothing was changed"),
            "expired": ("Approval expired", TraceStatus.REJECTED, "Nothing was changed"),
        }.get(status, ("Approved by a human", TraceStatus.COMPLETED, None))
        trace.add(
            TraceEventKind.APPROVAL,
            decided[0],
            decided[1],
            detail=decided[2],
            action_type=action.get("action_type"),
            action_status=status,
        )
        if status in ("succeeded", "failed", "executing", "approved"):
            ok = status == "succeeded"
            trace.add(
                TraceEventKind.ACTION_EXECUTION,
                "Deterministic execution",
                TraceStatus.COMPLETED if ok else TraceStatus.FAILED,
                detail=(
                    "Executed once by application code; audit event recorded"
                    if ok
                    else f"Not completed ({action.get('failure_code') or status})"
                ),
                action_type=action.get("action_type"),
                action_status=status,
                failure_code=action.get("failure_code"),
            )

    trace.add(
        TraceEventKind.RESPONSE,
        "Outcome recorded" if action is not None else "Response generated",
        model_calls=model_calls,
        tool_calls=len(tool_calls),
        citations=len(values.get("citations", [])),
        duration_ms=duration_ms,
    )
    return trace.events
