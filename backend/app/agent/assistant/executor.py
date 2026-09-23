"""Safe executor for model-requested tool calls.

The registry passed in (``build_commerce_tools()``) IS the capability allowlist: calls are
looked up by exact name in a dict built from it. No imports, getattr or reflection on
model-supplied names. Execution goes through the real Step 3 tool (its schema validation,
envelope and per-tool logging) with the trusted AgentContext injected as ToolRuntime.
"""

import json
import re
import time
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import BaseTool

from app.agent.assistant.result import ArgumentValue, ToolCallSummary, ToolOutcome
from app.agent.context import AgentContext
from app.agent.tools import envelope
from app.agent.tools.envelope import ErrorCode
from app.agent.tools.invoke import make_runtime

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_]{1,64}$")
_MAX_ARG_CHARS = 64


def safe_name(value: Any) -> str | None:
    return value if isinstance(value, str) and _SAFE_NAME.fullmatch(value) else None


class ToolExecutionError(Exception):
    """Host-side protocol violation (e.g. tool result lost its call id). Not model input."""


class ToolExecutor:
    def __init__(self, tools: Sequence[BaseTool]) -> None:
        registry: dict[str, BaseTool] = {}
        for tool in tools:
            if tool.name in registry:
                raise ValueError(f"duplicate tool name: {tool.name}")
            registry[tool.name] = tool
        self._registry = registry
        self._tools = tuple(tools)

    @property
    def tools(self) -> tuple[BaseTool, ...]:
        """Exactly the tools bound to the model."""
        return self._tools

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._registry)

    def execute(
        self, call: ToolCall, context: AgentContext, round_no: int
    ) -> tuple[ToolMessage, ToolCallSummary]:
        started = time.perf_counter()
        call_id = call["id"]
        name = call.get("name")
        args = call.get("args")
        tool = self._registry.get(name) if isinstance(name, str) else None

        if tool is None:
            shown = safe_name(name) or "<invalid>"
            result = envelope.failure(
                ErrorCode.UNKNOWN_TOOL,
                f"Tool '{shown}' is not available. Use only the listed tools.",
            )
            return self._message(result, call_id, shown, error=True), self._summary(
                round_no, shown, None, args, result, started
            )

        if not isinstance(args, dict):
            result = envelope.failure(
                ErrorCode.INVALID_ARGUMENTS, "Invalid arguments: arguments must be an object."
            )
            return self._message(result, call_id, tool.name, error=True), self._summary(
                round_no, tool.name, tool, {}, result, started
            )
        if "runtime" in args:
            # The runtime is host-injected; a model-supplied one is rejected, never merged.
            result = envelope.failure(
                ErrorCode.INVALID_ARGUMENTS, "Invalid arguments: runtime (extra_forbidden)."
            )
            return self._message(result, call_id, tool.name, error=True), self._summary(
                round_no, tool.name, tool, args, result, started
            )

        message = tool.invoke(
            {
                "type": "tool_call",
                "id": call_id,
                "name": tool.name,
                "args": {**args, "runtime": make_runtime(context)},
            }
        )
        if not isinstance(message, ToolMessage) or message.tool_call_id != call_id:
            raise ToolExecutionError("tool result is not a ToolMessage for the requested call")
        result = _parse_envelope(message.content)
        return message, self._summary(round_no, tool.name, tool, args, result, started)

    @staticmethod
    def _message(result: dict[str, Any], call_id: str, name: str, *, error: bool) -> ToolMessage:
        return ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            tool_call_id=call_id,
            name=name,
            status="error" if error else "success",
        )

    @staticmethod
    def _summary(
        round_no: int,
        name: str,
        tool: BaseTool | None,
        args: Any,
        result: dict[str, Any],
        started: float,
    ) -> ToolCallSummary:
        allowed = set(tool.tool_call_schema.model_fields) if tool is not None else set()
        given = args if isinstance(args, dict) else {}
        arguments = {k: _sanitise(v) for k, v in given.items() if k in allowed}
        rejected = sorted(
            n for n in (safe_name(k) or "<invalid>" for k in given if k not in allowed)
        )[:10]
        return ToolCallSummary(
            round=round_no,
            tool=name,
            arguments=arguments,
            rejected_argument_names=rejected if tool is not None else [],
            outcome=_outcome(result),
            error_code=None if result.get("ok") else result.get("error", {}).get("code"),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )


def _sanitise(value: Any) -> ArgumentValue:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:_MAX_ARG_CHARS]
    return "<omitted>"


def _parse_envelope(content: Any) -> dict[str, Any]:
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("ok"), bool):
        return data
    return envelope.failure(ErrorCode.INTERNAL_ERROR)


def _outcome(result: dict[str, Any]) -> ToolOutcome:
    if result.get("ok"):
        return "success"
    code = str(result.get("error", {}).get("code", ""))
    if code.endswith("_not_found"):
        return "not_found"
    if code in ("invalid_arguments", "unknown_tool", "service_unavailable"):
        return code  # type: ignore[return-value]
    return "internal_error"
