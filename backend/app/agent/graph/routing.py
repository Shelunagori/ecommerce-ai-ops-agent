"""Decision logic after a model step, and the graph's conditional-edge functions.

LangGraph conditional-edge functions cannot write state, but a rejected turn must be
recorded (error + invalid-call summaries). So the checks live in the pure function
``evaluate_model_turn``; the MODEL node calls it and records the decision, and
``route_after_model`` only reads what was recorded.

This is written independently of the Step-5 ``CommerceAssistant`` loop (the parity
oracle); only lower-level pieces are shared: the final-answer guard, limits and models.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END

from app.agent.assistant.answer import detect_protocol_artifact
from app.agent.assistant.executor import safe_name
from app.agent.assistant.limits import AssistantLimits
from app.agent.assistant.result import InvalidToolCallSummary
from app.agent.graph.state import CommerceGraphState

MODEL = "model"
TOOLS = "tools"
RETRIEVE = "retrieve"


@dataclass(frozen=True)
class TurnDecision:
    kind: Literal["tools", "retrieve", "answer", "error"]
    answer: str | None = None
    error_code: str | None = None
    error_detail: str | None = None
    invalid_calls: tuple[InvalidToolCallSummary, ...] = ()
    new_call_ids: tuple[str, ...] = field(default=())


def evaluate_model_turn(
    ai: AIMessage,
    *,
    round_no: int,
    seen_call_ids: Sequence[str],
    tool_calls_so_far: int,
    limits: AssistantLimits,
    retrieval_name: str | None = None,
    retrieval_done: bool = False,
) -> TurnDecision:
    """``tool_calls_so_far`` counts EVERY capability call of the run (commerce + policy
    retrieval, including invalid attempts). ``retrieval_name`` is the policy capability's
    name when it is enabled (None: Step-5 parity profile, every call is a commerce call).
    ``retrieval_done``: policy content was already returned to the model in this run."""
    # 1. Provider could not parse a tool request: execute nothing, fabricate nothing.
    if ai.invalid_tool_calls:
        return TurnDecision(
            kind="error",
            error_code="agent_protocol_error",
            error_detail="invalid_tool_calls",
            invalid_calls=tuple(
                InvalidToolCallSummary(
                    round=round_no, name=safe_name(c.get("name")), reason="unparseable"
                )
                for c in ai.invalid_tool_calls
            ),
        )

    calls = list(ai.tool_calls)
    # 2. Parsed tool calls. Visible text may legitimately be empty here.
    if calls:
        seen = set(seen_call_ids)
        new_ids: list[str] = []
        for call in calls:
            call_id = call.get("id")
            if not call_id or call_id in seen:
                return TurnDecision(
                    kind="error",
                    error_code="agent_protocol_error",
                    error_detail="tool_call_id",
                    invalid_calls=(
                        InvalidToolCallSummary(
                            round=round_no,
                            name=safe_name(call.get("name")),
                            reason="missing_id" if not call_id else "duplicate_id",
                        ),
                    ),
                )
            seen.add(call_id)
            new_ids.append(call_id)
        retrievals = [c for c in calls if retrieval_name and c.get("name") == retrieval_name]
        kind: Literal["tools", "retrieve"] = "retrieve" if retrievals else "tools"
        if retrievals and len(retrievals) != len(calls):
            # One AIMessage may not mix capability classes: execute nothing of it.
            return _protocol("mixed_capability_batch")
        if kind == "tools" and retrieval_done:
            # Deterministic boundary: once retrieved (untrusted) policy text is in the model's
            # context, no new commerce capability may run in this user turn.
            return _protocol("commerce_call_after_retrieval")
        if len(retrievals) > 1:
            return _limit("max_retrievals_per_turn")
        # Whole-batch budgets, checked before anything in the batch executes.
        if len(calls) > limits.max_tool_calls_per_turn:
            return _limit("max_tool_calls_per_turn")
        if tool_calls_so_far + len(calls) > limits.max_tool_calls:
            return _limit("max_tool_calls")
        if round_no >= limits.max_model_rounds:
            # No model round left to read the results: reject before execution.
            return _limit("max_model_rounds")
        return TurnDecision(kind=kind, new_call_ids=tuple(new_ids))

    # 3. No tool calls: the visible text must be a real answer.
    answer = ai.text.strip()
    if not answer:
        return TurnDecision(kind="error", error_code="agent_empty_answer")
    artifact = detect_protocol_artifact(answer)
    if artifact is not None:
        return TurnDecision(
            kind="error",
            error_code="agent_protocol_error",
            error_detail="protocol_artifact",
            invalid_calls=(
                InvalidToolCallSummary(round=round_no, name=artifact.name, reason=artifact.kind),
            ),
        )
    return TurnDecision(kind="answer", answer=answer)


def _protocol(detail: str) -> TurnDecision:
    return TurnDecision(kind="error", error_code="agent_protocol_error", error_detail=detail)


def _limit(detail: str) -> TurnDecision:
    return TurnDecision(kind="error", error_code="agent_limit_exceeded", error_detail=detail)


def route_after_model(state: CommerceGraphState) -> str:
    """Approved commerce batch -> TOOLS; approved policy retrieval -> RETRIEVE;
    final answer or terminal error -> END."""
    if state.get("error") is None and state.get("pending") is not None:
        return RETRIEVE if state.get("pending_kind") == "retrieval" else TOOLS
    return END


def route_after_tools(state: CommerceGraphState) -> str:
    """Complete batch -> MODEL; host-side tool protocol failure -> END."""
    return END if state.get("error") is not None else MODEL


def route_after_retrieve(state: CommerceGraphState) -> str:
    """Retrieval ToolMessage appended -> MODEL; retrieval infrastructure failure -> END
    (the model is never asked to answer after a failed retrieval)."""
    return END if state.get("error") is not None else MODEL
