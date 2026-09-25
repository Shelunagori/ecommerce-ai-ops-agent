"""OPT-IN live Cloudflare Workers AI protocol check (compatibility, not answer quality).

    RUN_CLOUDFLARE_INTEGRATION=1 CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... \\
      [CLOUDFLARE_MODEL=...] uv run pytest tests/llm/test_cloudflare_live.py

Skipped by default and in CI. Synthetic text only. Verifies that the configured model
returns a STRUCTURED tool call (not JSON written as text) and completes a tool round trip.

``test_live_raw_tool_call_shape`` prints the STRUCTURE of the raw Workers AI tool call (key
names, the id's type/emptiness, the arguments' type) - never values, prompts or the token -
so a missing/empty provider id can be confirmed against production (run with ``-s``).
"""

import json
import os

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agent.assistant.answer import detect_protocol_artifact
from app.agent.llm import get_llm_provider
from app.agent.llm.config import LLMConfig, cloudflare_base_url
from app.agent.llm.factory import CORRELATION_ID_PREFIX
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
    assert first.invalid_tool_calls == []
    assert isinstance(call["id"], str) and call["id"].strip()  # usable (provider or generated)
    source = "generated" if call["id"].startswith(CORRELATION_ID_PREFIX) else "provider"
    print(f"\ntool_call_id source: {source}")
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


def test_live_raw_tool_call_shape():
    """Phase-1 evidence: the raw provider shape, structure only (no values are printed)."""
    config = LLMConfig.from_settings(Settings(llm_provider="cloudflare"), provider="cloudflare")
    [tool] = [t for t in build_commerce_tools() if t.name == "get_order"]
    token = config.cloudflare_api_token.get_secret_value()  # type: ignore[union-attr]
    r = httpx.post(
        f"{cloudflare_base_url(config.cloudflare_account_id)}/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": config.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "What is the status of order ORD-1001?"},
            ],
            "tools": [convert_to_openai_tool(tool)],
            "max_tokens": 256,
            "temperature": 0,
        },
        timeout=60,
    )
    assert r.status_code == 200, r.status_code
    message = r.json()["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    shape = [
        {
            "keys": sorted(c),
            "id_type": type(c.get("id")).__name__,
            "id_empty": not (isinstance(c.get("id"), str) and c["id"].strip()),
            "function_keys": sorted(c.get("function") or {}),
            "arguments_type": type((c.get("function") or {}).get("arguments")).__name__,
        }
        for c in calls
    ]
    print("\nraw tool_call shape:", json.dumps(shape))
    assert calls, "expected a structured tool call"
