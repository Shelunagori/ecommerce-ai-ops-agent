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


@dataclass(frozen=True)
class TurnDecision:
    kind: Literal["tools", "answer", "error"]
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
) -> TurnDecision:
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
        # Whole-batch budgets, checked before anything in the batch executes.
        if len(calls) > limits.max_tool_calls_per_turn:
            return _limit("max_tool_calls_per_turn")
        if tool_calls_so_far + len(calls) > limits.max_tool_calls:
            return _limit("max_tool_calls")
        if round_no >= limits.max_model_rounds:
            # No model round left to read the results: reject before execution.
            return _limit("max_model_rounds")
        return TurnDecision(kind="tools", new_call_ids=tuple(new_ids))

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


def _limit(detail: str) -> TurnDecision:
    return TurnDecision(kind="error", error_code="agent_limit_exceeded", error_detail=detail)


def route_after_model(state: CommerceGraphState) -> str:
    """Approved batch -> TOOLS; final answer or terminal error -> END."""
    if state.get("error") is None and state.get("pending") is not None:
        return TOOLS
    return END


def route_after_tools(state: CommerceGraphState) -> str:
    """Complete batch -> MODEL; host-side tool protocol failure -> END."""
    return END if state.get("error") is not None else MODEL
