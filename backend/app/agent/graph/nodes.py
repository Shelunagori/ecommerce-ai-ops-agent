"""MODEL and TOOLS nodes. Each returns only a state update.

External side effects: MODEL invokes the Step-4 provider; TOOLS executes the read-only
Step-3 tools through the hardened ``ToolExecutor`` (registry allowlist, trusted runtime
injection, argument checks, sanitised summaries, call-id checks, Step-3 logging).
"""

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.agent.assistant.executor import ToolExecutionError, ToolExecutor
from app.agent.assistant.limits import AssistantLimits
from app.agent.context import AgentContext
from app.agent.graph.routing import evaluate_model_turn
from app.agent.graph.state import CommerceGraphState, GraphError, scope_digest
from app.agent.llm import LLMError, LLMProvider
from app.agent.prompts import assistant as assistant_prompt

THREAD_SCOPE_MESSAGE = "This conversation thread belongs to a different account."


def _trusted_context(runtime: Runtime[AgentContext]) -> AgentContext:
    context = runtime.context
    if not isinstance(context, AgentContext):
        raise TypeError("graph runtime context must be a validated AgentContext")
    return context


def _error(code: str, detail: str | None = None, message: str | None = None) -> GraphError:
    return {"code": code, "message": message, "detail": detail}


class GraphNodes:
    def __init__(
        self, provider: LLMProvider, executor: ToolExecutor, limits: AssistantLimits
    ) -> None:
        self._provider = provider
        self._executor = executor
        self._limits = limits

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
        try:
            chat = self._provider.invoke_chat(
                state["messages"],
                tools=self._executor.tools,
                operation=assistant_prompt.PROMPT_ID,
                prompt_version=assistant_prompt.PROMPT_VERSION,
            )
        except LLMError as exc:
            return {**update, "pending": None, "error": _error(exc.code, "llm", exc.message)}
        ai: AIMessage = chat.message  # the provider's original message, never rebuilt

        decision = evaluate_model_turn(
            ai,
            round_no=round_no,
            seen_call_ids=state.get("seen_tool_call_ids", []),
            tool_calls_so_far=len(state.get("tool_calls", [])),
            limits=self._limits,
        )
        if decision.kind == "tools":
            seen = [*state.get("seen_tool_call_ids", []), *decision.new_call_ids]
            return {**update, "pending": ai, "seen_tool_call_ids": seen}
        if decision.kind == "answer":
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

    def tools(self, state: CommerceGraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        context = _trusted_context(runtime)
        ai = state.get("pending")
        if ai is None:
            raise RuntimeError("tools node reached without an approved tool batch")
        round_no = state.get("model_calls", 0)
        summaries = list(state.get("tool_calls", []))
        tool_messages = []
        for call in ai.tool_calls:  # sequential, in request order; no model call in between
            try:
                message, summary = self._executor.execute(call, context, round_no)
            except ToolExecutionError:
                return {
                    "pending": None,
                    "tool_calls": summaries,
                    "error": _error("agent_protocol_error", "tool_result"),
                }
            tool_messages.append(message)
            summaries.append(summary.model_dump(mode="json"))
        # One update: the original AIMessage, then its complete ordered ToolMessage batch.
        return {"messages": [ai, *tool_messages], "pending": None, "tool_calls": summaries}
