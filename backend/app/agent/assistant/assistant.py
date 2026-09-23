"""CommerceAssistant: an explicit, bounded, sequential tool-calling loop.

    messages = [system, user]
    for round in 1..max_model_rounds:
        ai = model(messages, tools=registry)            # Step 4 provider, per-call retry
        invalid_tool_calls  -> agent_protocol_error     (nothing executed, nothing fabricated)
        no tool calls       -> final answer (ai.text)   (empty -> agent_empty_answer;
                               textual pseudo tool call / bare {} -> agent_protocol_error)
        missing/duplicate id-> agent_protocol_error
        batch over budget   -> agent_limit_exceeded     (whole batch rejected, none executed)
        append ai (the ORIGINAL AIMessage, provider metadata intact)
        execute every call of the batch sequentially, append ToolMessages in call order
    rounds exhausted        -> agent_limit_exceeded

The host never retries tool calls; a not-found or invalid-arguments result is fed back to
the model as a business outcome. Only the provider layer retries transient model failures.
"""

import logging
import time
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from app.agent.assistant.answer import detect_protocol_artifact
from app.agent.assistant.errors import AssistantError
from app.agent.assistant.executor import ToolExecutionError, ToolExecutor, safe_name
from app.agent.assistant.limits import AssistantLimits
from app.agent.assistant.result import AssistantResult, InvalidToolCallSummary, ToolCallSummary
from app.agent.context import AgentContext
from app.agent.llm import LLMError, LLMProvider
from app.agent.prompts import assistant as assistant_prompt
from app.agent.tools import build_commerce_tools

logger = logging.getLogger("app.agent.assistant")

MAX_INPUT_CHARS = 4000


class CommerceAssistant:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        tools: Sequence[BaseTool] | None = None,
        limits: AssistantLimits | None = None,
    ) -> None:
        self._provider = provider
        self._executor = ToolExecutor(tools if tools is not None else build_commerce_tools())
        self._limits = limits or AssistantLimits.from_settings()

    @property
    def bound_tool_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self._executor.tools)

    def run(self, text: str, context: AgentContext) -> AssistantResult:
        if not isinstance(context, AgentContext):
            raise TypeError("context must be a validated AgentContext")
        started = time.perf_counter()
        run = _RunState()
        try:
            result = self._run(text, context, run, started)
        except AssistantError as exc:
            exc.model_calls = run.model_calls
            exc.tool_calls = list(run.tool_calls)
            exc.invalid_tool_calls = list(run.invalid_calls)
            self._log(context, run, exc.code, started, exc.detail)
            raise
        self._log(context, run, "ok", started)
        return result

    def _run(
        self, text: str, context: AgentContext, run: "_RunState", started: float
    ) -> AssistantResult:
        cleaned = text.strip() if isinstance(text, str) else ""
        if not cleaned or len(cleaned) > MAX_INPUT_CHARS:
            raise AssistantError("agent_input_invalid")

        limits = self._limits
        messages: list[BaseMessage] = assistant_prompt.build_messages(cleaned)
        seen_ids: set[str] = set()

        for round_no in range(1, limits.max_model_rounds + 1):
            ai = self._call_model(messages, run)

            if ai.invalid_tool_calls:
                run.invalid_calls.extend(
                    InvalidToolCallSummary(
                        round=round_no, name=safe_name(c.get("name")), reason="unparseable"
                    )
                    for c in ai.invalid_tool_calls
                )
                raise AssistantError("agent_protocol_error", detail="invalid_tool_calls")

            calls = list(ai.tool_calls)
            if not calls:
                answer = ai.text.strip()
                if not answer:
                    raise AssistantError("agent_empty_answer")
                artifact = detect_protocol_artifact(answer)
                if artifact is not None:
                    # Text-only pseudo tool call or bare {} / []: never executed, never
                    # returned as an answer. Tools run only from parsed tool_calls.
                    run.invalid_calls.append(
                        InvalidToolCallSummary(
                            round=round_no, name=artifact.name, reason=artifact.kind
                        )
                    )
                    raise AssistantError("agent_protocol_error", detail="protocol_artifact")
                return AssistantResult(
                    answer=answer,
                    provider=self._provider.info.provider,
                    model=self._provider.info.model,
                    prompt_version=assistant_prompt.PROMPT_VERSION,
                    model_calls=run.model_calls,
                    tool_calls=list(run.tool_calls),
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )

            for call in calls:
                call_id = call.get("id")
                if not call_id or call_id in seen_ids:
                    run.invalid_calls.append(
                        InvalidToolCallSummary(
                            round=round_no,
                            name=safe_name(call.get("name")),
                            reason="missing_id" if not call_id else "duplicate_id",
                        )
                    )
                    raise AssistantError("agent_protocol_error", detail="tool_call_id")
                seen_ids.add(call_id)

            # Budgets are checked for the WHOLE batch before anything in it executes.
            if len(calls) > limits.max_tool_calls_per_turn:
                raise AssistantError("agent_limit_exceeded", detail="max_tool_calls_per_turn")
            if len(run.tool_calls) + len(calls) > limits.max_tool_calls:
                raise AssistantError("agent_limit_exceeded", detail="max_tool_calls")
            if round_no == limits.max_model_rounds:
                # Results could never be read by the model again.
                raise AssistantError("agent_limit_exceeded", detail="max_model_rounds")

            messages.append(ai)  # original AIMessage, unmodified (keeps provider metadata)
            for call in calls:  # sequential by design (Step 5)
                try:
                    tool_message, summary = self._executor.execute(call, context, round_no)
                except ToolExecutionError:
                    raise AssistantError("agent_protocol_error", detail="tool_result") from None
                messages.append(tool_message)
                run.tool_calls.append(summary)

        raise AssistantError("agent_limit_exceeded", detail="max_model_rounds")  # pragma: no cover

    def _call_model(self, messages: list[BaseMessage], run: "_RunState") -> AIMessage:
        try:
            chat = self._provider.invoke_chat(
                messages,
                tools=self._executor.tools,
                operation=assistant_prompt.PROMPT_ID,
                prompt_version=assistant_prompt.PROMPT_VERSION,
            )
        except LLMError as exc:
            run.model_calls += 1
            raise AssistantError(exc.code, exc.message, detail="llm") from None
        run.model_calls += 1
        return chat.message

    def _log(
        self,
        context: AgentContext,
        run: "_RunState",
        outcome: str,
        started: float,
        detail: str | None = None,
    ) -> None:
        # Identifiers and counts only: never the user message, prompts, model output or
        # tool payloads. tenant_id stays in trusted server logs (as in Step 3).
        fields = {
            "provider": self._provider.info.provider,
            "model": self._provider.info.model,
            "prompt_version": assistant_prompt.PROMPT_VERSION,
            "tenant_id": str(context.tenant_id),
            "model_calls": run.model_calls,
            "tool_call_count": len(run.tool_calls),
            "tool_names": [c.tool for c in run.tool_calls],
            "invalid_tool_calls": len(run.invalid_calls),
            "outcome": outcome,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if context.request_id:
            fields["request_id"] = context.request_id
        if detail:
            fields["detail"] = detail
        logger.log(
            logging.INFO if outcome == "ok" else logging.WARNING, "assistant run", extra=fields
        )


class _RunState:
    def __init__(self) -> None:
        self.model_calls = 0
        self.tool_calls: list[ToolCallSummary] = []
        self.invalid_calls: list[InvalidToolCallSummary] = []
