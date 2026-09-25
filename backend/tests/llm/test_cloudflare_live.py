"""OPT-IN live Cloudflare Workers AI protocol check (compatibility, not answer quality).

    RUN_CLOUDFLARE_INTEGRATION=1 CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... \\
      [CLOUDFLARE_MODEL=...] uv run pytest tests/llm/test_cloudflare_live.py

Skipped by default and in CI. Synthetic text only. Verifies that the configured model
returns a STRUCTURED tool call (not JSON written as text) and completes a tool round trip.
"""

import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.agent.assistant.answer import detect_protocol_artifact
from app.agent.llm import get_llm_provider
from app.agent.tools import build_commerce_tools
from app.core.config import Settings

pytestmark = [
    pytest.mark.llm_integration,
    pytest.mark.skipif(
        os.getenv("RUN_CLOUDFLARE_INTEGRATION") != "1", reason="set RUN_CLOUDFLARE_INTEGRATION=1"
    ),
]

SYSTEM = (
    "You are an ecommerce operations assistant. Use the provided tools for order facts. "
    "Never invent data."
)


def test_live_tool_call_round_trip():
    provider = get_llm_provider(Settings(llm_provider="cloudflare"), provider="cloudflare")
    tools = [t for t in build_commerce_tools() if t.name == "get_order"]
    messages = [SystemMessage(SYSTEM), HumanMessage("What is the status of order ORD-1001?")]
    first = provider.invoke_chat(messages, tools=tools, operation="live_check").message
    assert first.tool_calls, "expected a structured tool call"
    call = first.tool_calls[0]
    assert call["name"] == "get_order" and call["args"].get("order_number") == "ORD-1001"
    assert call["id"]
    result = ToolMessage(
        content='{"ok": true, "data": {"order_number": "ORD-1001", "status": "delivered"}}',
        tool_call_id=call["id"],
        name="get_order",
    )
    second = provider.invoke_chat(
        [*messages, first, result], tools=tools, operation="live_check"
    ).message
    assert not second.tool_calls and second.text.strip()
    assert detect_protocol_artifact(second.text) is None
