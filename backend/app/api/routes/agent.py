"""Agent API (Phase 6): chat, approvals, sanitised history, identity.

Tenant and user come ONLY from the trusted principal (``get_principal``): never from the
request body and never from the model. Approvals use action RESOURCE ids plus the arguments
hash the human saw; there are no model-generated execution tokens. Raw LangGraph state,
prompts, tool payloads, retrieved content and vectors are never returned.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, Path, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from app.actions import errors as action_errors
from app.agent.assistant import AssistantError
from app.agent.assistant.result import (
    ActionSummary,
    AssistantResult,
    PolicyCitation,
    RetrievalSummary,
    ToolCallSummary,
)
from app.agent.context import AgentContext
from app.agent.trace import ExecutionTraceEvent
from app.api.agent_runtime import internal_thread_id, runtime_for
from app.api.deps import CurrentPrincipal, ReadSession, get_app_settings, get_jwt_verifier
from app.api.ratelimit import charge_message
from app.auth.errors import PublicDemoReadOnlyError
from app.auth.principal import identity_only
from app.core.request_context import request_id_var

router = APIRouter(prefix="/api", tags=["agent"])

ThreadId = Field(pattern=r"^[A-Za-z0-9_-]{1,48}$", description="Client-chosen conversation id")


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)
    thread_id: str = ThreadId


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Approve exactly what was shown: the hash from the pending action card.
    arguments_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class AgentResponse(BaseModel):
    thread_id: str
    answer: str
    prompt_version: str
    model_calls: int
    duration_ms: float
    tool_calls: list[ToolCallSummary]
    retrievals: list[RetrievalSummary]
    citations: list[PolicyCitation]
    action: ActionSummary | None = None
    # Ordered, safe record of what this run did (graph nodes, tools, retrieval, grounding,
    # approval). Response-only: not stored with the conversation history.
    execution_trace: list[ExecutionTraceEvent] = []


class ActionOut(BaseModel):
    id: str
    action_type: str
    status: str
    summary: str
    arguments: dict[str, Any]
    arguments_hash: str
    evidence: list[dict[str, Any]]
    created_at: str
    expires_at: str
    decided_at: str | None
    completed_at: str | None
    result: dict[str, Any] | None
    failure_code: str | None
    events: list[dict[str, Any]] = []


class DecisionOut(BaseModel):
    action: ActionOut
    answer: str | None = None  # the deterministic outcome message when the graph was resumed
    # Trace of the resumed run (full run incl. the decision); None when no graph was resumed.
    execution_trace: list[ExecutionTraceEvent] | None = None


class MembershipOut(BaseModel):
    tenant_id: str
    slug: str
    name: str
    role: Literal["member", "approver"]


class MeOut(BaseModel):
    subject: str
    auth_mode: str
    memberships: list[MembershipOut]
    public_demo: bool = False  # verified anonymous visitor: read-only demo tenant


class HistoryMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str


class HistoryOut(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]
    pending_action: ActionOut | None = None


def _context(principal: Any, _request: Request) -> AgentContext:
    """Trusted runtime context: tenant from the principal, request id from the middleware."""
    return AgentContext(principal.tenant.tenant_id, request_id_var.get())


def _response(thread_id: str, r: AssistantResult) -> AgentResponse:
    return AgentResponse(
        thread_id=thread_id,
        answer=r.answer,
        prompt_version=r.prompt_version,
        model_calls=r.model_calls,
        duration_ms=r.duration_ms,
        tool_calls=r.tool_calls,
        retrievals=r.retrievals,
        citations=r.citations,
        action=r.action,
        execution_trace=r.execution_trace,
    )


def _require_actions(principal: Any) -> None:
    """Action resources do not exist for the read-only public demo (not just hidden in UI)."""
    if not principal.can_use_actions:
        raise PublicDemoReadOnlyError()


def _action_out(view: Any, events: list[dict[str, Any]] | None = None) -> ActionOut:
    d = view.as_dict()
    return ActionOut(**{k: v for k, v in d.items()}, events=events or [])


@router.get("/me", response_model=MeOut)
def me(request: Request, session: ReadSession) -> MeOut:
    authorization = request.headers.get("Authorization")
    identity = identity_only(
        session,
        get_app_settings(request),
        authorization=authorization,
        verifier=get_jwt_verifier(request),
    )
    return MeOut(
        subject=identity.subject,
        auth_mode=get_app_settings(request).auth_mode,
        memberships=[
            MembershipOut(tenant_id=str(m.tenant_id), slug=m.slug, name=m.name, role=m.role)
            for m in identity.memberships
        ],
        public_demo=identity.public_demo,
    )


@router.post("/agent/messages", response_model=AgentResponse)
def post_message(body: MessageIn, principal: CurrentPrincipal, request: Request) -> AgentResponse:
    # Before any model call: one user cannot exhaust the hosted model budget.
    charge_message(request.app, principal)
    runtime = runtime_for(request.app)
    assistant = runtime.assistant_for(principal)  # read-only graph for the public demo
    context = _context(principal, request)
    thread = internal_thread_id(principal.subject, body.thread_id)
    result = assistant.run(body.text, context, thread_id=thread)
    return _response(body.thread_id, result)


@router.get("/agent/threads/{thread_id}/messages", response_model=HistoryOut)
def thread_history(
    principal: CurrentPrincipal,
    request: Request,
    thread_id: str = Path(pattern=r"^[A-Za-z0-9_-]{1,48}$"),
) -> HistoryOut:
    runtime = runtime_for(request.app)
    assistant = runtime.assistant_for(principal)
    context = _context(principal, request)
    thread = internal_thread_id(principal.subject, thread_id)
    messages = [HistoryMessage(**m) for m in assistant.history(context, thread)]
    pending = assistant.pending_approval(context, thread) if principal.can_use_actions else None
    pending_out = None
    if pending is not None:
        pending_out = _action_out(runtime.actions.get(principal.tenant, uuid.UUID(pending["id"])))
    return HistoryOut(thread_id=thread_id, messages=messages, pending_action=pending_out)


@router.get("/agent/actions", response_model=list[ActionOut])
def list_actions(
    principal: CurrentPrincipal,
    request: Request,
    status: Literal[
        "pending_approval", "approved", "rejected", "executing", "succeeded", "failed", "expired"
    ]
    | None = Query(default=None),
) -> list[ActionOut]:
    _require_actions(principal)
    views = runtime_for(request.app).actions.list_requests(principal.tenant, status=status)
    return [_action_out(v) for v in views]


@router.get("/agent/actions/{action_id}", response_model=ActionOut)
def get_action(action_id: uuid.UUID, principal: CurrentPrincipal, request: Request) -> ActionOut:
    _require_actions(principal)
    actions = runtime_for(request.app).actions
    view = actions.get(principal.tenant, action_id)
    return _action_out(view, actions.events(principal.tenant, action_id))


def _decide(
    action_id: uuid.UUID,
    decision: Literal["approve", "reject"],
    body: DecisionIn,
    principal: Any,
    request: Request,
) -> DecisionOut:
    _require_actions(principal)
    if not principal.can_approve:
        raise action_errors.ActionForbiddenError()
    runtime = runtime_for(request.app)
    context = _context(principal, request)
    view = runtime.actions.get(principal.tenant, action_id)  # tenant-scoped: other tenant -> 404
    pending = (
        runtime.assistant.pending_approval(context, thread_key=view.thread_key)
        if view.thread_key
        else None
    )
    if pending is not None and pending.get("id") == str(action_id):
        result = runtime.assistant.resume(
            context,
            thread_key=view.thread_key,
            action_id=action_id,
            decision=decision,
            decided_by=principal.subject,
            expected_hash=body.arguments_hash,
        )
        final = runtime.actions.get(principal.tenant, action_id)
        return DecisionOut(
            action=_action_out(final, runtime.actions.events(principal.tenant, action_id)),
            answer=result.answer,
            execution_trace=result.execution_trace,
        )
    # No paused graph (already resumed, or resumed but execution did not finish):
    # idempotent decision, and a SAFE retry of an approved-but-unexecuted request.
    try:
        decided = runtime.actions.decide(
            principal.tenant,
            action_id,
            decision,
            decided_by=principal.subject,
            expected_hash=body.arguments_hash,
        )
    except action_errors.ApprovalExpiredError:
        decided = runtime.actions.get(principal.tenant, action_id)
    if decision == "approve" and decided.status == "approved":
        try:
            decided = runtime.actions.execute(principal.tenant, action_id)
        except action_errors.ActionError:
            decided = runtime.actions.get(principal.tenant, action_id)
    return DecisionOut(
        action=_action_out(decided, runtime.actions.events(principal.tenant, action_id))
    )


@router.post("/agent/actions/{action_id}/approve", response_model=DecisionOut)
def approve(
    action_id: uuid.UUID, body: DecisionIn, principal: CurrentPrincipal, request: Request
) -> DecisionOut:
    return _decide(action_id, "approve", body, principal, request)


@router.post("/agent/actions/{action_id}/reject", response_model=DecisionOut)
def reject(
    action_id: uuid.UUID, body: DecisionIn, principal: CurrentPrincipal, request: Request
) -> DecisionOut:
    return _decide(action_id, "reject", body, principal, request)


# --- AssistantError -> HTTP ---------------------------------------------------------------------
ASSISTANT_STATUS = {
    "agent_input_invalid": 400,
    "agent_approval_pending": 409,
    "agent_no_pending_approval": 409,
    "agent_thread_conflict": 409,
    "agent_limit_exceeded": 422,
    "agent_protocol_error": 502,
    "agent_empty_answer": 502,
    "agent_grounding_error": 502,
    "agent_retrieval_error": 503,
    "agent_action_error": 503,
    "agent_state_unavailable": 503,
    "llm_timeout": 504,
    "llm_auth_failed": 502,
}


def assistant_status(exc: AssistantError) -> int:
    if exc.code in ASSISTANT_STATUS:
        return ASSISTANT_STATUS[exc.code]
    return 503 if exc.code.startswith("llm_") else 500
