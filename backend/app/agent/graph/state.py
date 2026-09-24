"""Graph state and thread/tenant scoping for the LangGraph commerce assistant (Step 6).

Checkpointed state holds only what graph execution and safe result construction need:
messages, per-run accounting and a terminal outcome. Never sessions, provider clients,
settings, secrets, raw tenant IDs or raw thread IDs. Tenant identity travels as LangGraph
runtime *context* (``AgentContext``), which is not checkpointed and never reaches the model.
"""

import hashlib
import re
import uuid
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langgraph.graph.message import add_messages

_SAFE_THREAD_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Per-run fields: the runner resets them at the start of EVERY user run, also on a
# continued thread. `messages` (add_messages reducer) and `scope_digest` are NOT reset.
RUN_RESET: dict[str, Any] = {
    "pending": None,
    "model_calls": 0,
    "tool_calls": [],
    "invalid_tool_calls": [],
    "seen_tool_call_ids": [],
    "answer": None,
    "error": None,
}


# Closes a checkpointed user turn that ended in an application/graph error, so a continued
# thread never has an unanswered user message. Stable and generic on purpose: no error codes,
# exception text, provider/database details or tenant data. Written by the runner, never by
# the model (it is not a model call and does not count as one).
FAILURE_MARKER_TEXT = "The previous request could not be completed."
FAILURE_MARKER_KEY = "commerceops_turn"  # response_metadata flag; not sent to providers


def failure_marker() -> AIMessage:
    return AIMessage(
        content=FAILURE_MARKER_TEXT,
        id=f"turn-failed-{uuid.uuid4().hex}",
        response_metadata={FAILURE_MARKER_KEY: "failed"},
    )


def is_failure_marker(message: BaseMessage) -> bool:
    return (
        isinstance(message, AIMessage)
        and message.response_metadata.get(FAILURE_MARKER_KEY) == "failed"
    )


def turn_is_open(messages: Sequence[BaseMessage]) -> bool:
    """True when history does not end with a closed assistant turn (a final AIMessage
    without tool calls - a real answer or a failure marker)."""
    if not messages:
        return False
    last = messages[-1]
    return not (isinstance(last, AIMessage) and not last.tool_calls)


class GraphError(TypedDict):
    code: str
    message: str | None  # safe message (LLM errors keep their Step-4 message)
    detail: str | None  # safe internal classification, e.g. "max_tool_calls"


class CommerceGraphState(TypedDict, total=False):
    # Conversation history. add_messages APPENDS node updates (never replaces history).
    messages: Annotated[list[AnyMessage], add_messages]
    # sha256 fingerprint of the trusted tenant bound to this thread (not the tenant ID).
    # Set by the first model step; a later run whose runtime context differs is refused.
    scope_digest: str
    # Original AIMessage of an approved tool batch, waiting for the TOOLS node. The tools
    # node appends it together with the complete ordered ToolMessage batch, then clears it.
    pending: AIMessage | None
    model_calls: int
    # Plain dicts (ToolCallSummary / InvalidToolCallSummary .model_dump()) so checkpoints
    # hold only JSON primitives, not application classes.
    tool_calls: list[dict[str, Any]]
    invalid_tool_calls: list[dict[str, Any]]
    seen_tool_call_ids: list[str]
    answer: str | None
    error: GraphError | None


def validate_thread_id(thread_id: Any) -> str:
    if not isinstance(thread_id, str) or not _SAFE_THREAD_ID.fullmatch(thread_id):
        raise ValueError("thread_id must be 1-64 chars of [A-Za-z0-9._-]")
    return thread_id


def checkpoint_thread_key(tenant_id: uuid.UUID, thread_id: str) -> str:
    """Internal checkpoint key: ``cg1-<sha256(tenant_id + ":" + thread_id)>``.

    The same caller thread ID under two tenants maps to two unrelated keys, so a reused
    thread name can never load another tenant's checkpoint.
    """
    if not isinstance(tenant_id, uuid.UUID):
        raise TypeError("tenant_id must be a uuid.UUID")
    digest = hashlib.sha256(f"{tenant_id}:{validate_thread_id(thread_id)}".encode()).hexdigest()
    return f"cg1-{digest}"


def scope_digest(tenant_id: uuid.UUID) -> str:
    if not isinstance(tenant_id, uuid.UUID):
        raise TypeError("tenant_id must be a uuid.UUID")
    return hashlib.sha256(f"cg1-scope:{tenant_id}".encode()).hexdigest()
