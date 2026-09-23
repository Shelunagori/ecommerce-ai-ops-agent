"""Invoke a registered tool directly (no model): used by the dev CLI and tests.

Mirrors what an agent runtime does: the host supplies ToolRuntime(context=AgentContext);
the caller supplies only business arguments.
"""

import json
from typing import Any

from langchain.tools import BaseTool, ToolRuntime

from app.agent.context import AgentContext
from app.agent.tools import envelope
from app.agent.tools.envelope import ErrorCode


def make_runtime(context: AgentContext) -> ToolRuntime[AgentContext]:
    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _chunk: None,
        tool_call_id=None,
        store=None,
    )


def invoke_tool(tool: BaseTool, args: dict[str, Any], context: AgentContext) -> dict[str, Any]:
    """Run ``tool`` with model-style ``args`` under a trusted ``context``; return the envelope."""
    if not isinstance(args, dict) or "runtime" in args:
        return envelope.failure(
            ErrorCode.INVALID_ARGUMENTS, "Invalid arguments: runtime (forbidden)."
        )
    return as_envelope(tool.invoke({**args, "runtime": make_runtime(context)}))


def as_envelope(result: Any) -> dict[str, Any]:
    """Tools return dicts; only the validation callback returns the envelope as JSON text."""
    if isinstance(result, str):
        return json.loads(result)
    if isinstance(result, dict):
        return result
    raise TypeError(f"unexpected tool result type: {type(result).__name__}")
