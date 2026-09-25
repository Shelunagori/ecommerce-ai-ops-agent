"""Cloudflare Workers AI chat provider (OpenAI-compatible endpoint), fully offline.

Every HTTP exchange goes through an ``httpx.MockTransport``: the tests see exactly what the
backend would send to Workers AI and feed back recorded-shape responses.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.actions.capability import action_tools
from app.agent.llm import LLMConfigurationError, LLMError, get_llm_provider
from app.agent.llm.config import LLMConfig, cloudflare_base_url
from app.agent.llm.factory import build_provider, openai_compatible_messages
from app.agent.llm.intent import IntentAnalysis
from app.agent.llm.provider import ChatModelProvider
from app.agent.rag.capability import policy_search_tool
from app.agent.tools import build_commerce_tools
from tests.llm.conftest import settings

ACCOUNT = "0123456789abcdef0123456789abcdef"
TOKEN = "cf-TEST-token-never-log-7c1d"
MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct"
TOOLS = build_commerce_tools()


def cf_settings(**over: Any):
    base = {
        "llm_provider": "cloudflare",
        "cloudflare_account_id": ACCOUNT,
        "cloudflare_api_token": TOKEN,
        "llm_max_retries": 1,
    }
    return settings(**{**base, **over})


def completion(message: dict[str, Any], finish: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def tool_call(call_id: str, name: str, **args: Any) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class Workers:
    """Scripted Workers AI endpoint: each item is a response or an exception to raise."""

    def __init__(self, *script: httpx.Response | Exception | Callable[[httpx.Request], Any]):
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step) and not isinstance(step, httpx.Response):
            return step(request)
        return step

    def body(self, i: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[i].content)


def provider(workers: Workers, **over: Any) -> ChatModelProvider:
    config = LLMConfig.from_settings(cf_settings(**over))
    p = build_provider(config, http_client=httpx.Client(transport=httpx.MockTransport(workers)))
    assert isinstance(p, ChatModelProvider)
    p._sleep = lambda _s: None  # no real back-off in tests
    return p


def ok(message: dict[str, Any], finish: str = "stop") -> httpx.Response:
    return httpx.Response(200, json=completion(message, finish))


def chat(p: ChatModelProvider, messages=None, tools=TOOLS):
    return p.invoke_chat(
        messages or [SystemMessage("You are a commerce assistant."), HumanMessage("Hi")],
        tools=tools,
        operation="commerce_assistant",
        prompt_version="commerce-assistant-v4",
    )


# --- 1-3: construction, endpoint, credentials ---------------------------------------------------
def test_construction_performs_no_network_io(no_network):
    p = get_llm_provider(cf_settings())
    assert (p.info.provider, p.info.model) == ("cloudflare", MODEL)
    model = p._chat_model
    assert isinstance(model, ChatOpenAI)
    assert model.max_retries == 0 and model.temperature == 0  # our RetryPolicy owns retries
    assert no_network == []


def test_endpoint_is_the_account_scoped_openai_compatible_url():
    w = Workers(ok({"role": "assistant", "content": "Hello!"}))
    chat(provider(w))
    req = w.requests[0]
    assert str(req.url) == (
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/v1/chat/completions"
    )
    assert req.method == "POST"
    body = w.body()
    assert body["model"] == MODEL and body["max_tokens"] == 1024 and body["stream"] is False
    assert cloudflare_base_url(ACCOUNT).endswith(f"/accounts/{ACCOUNT}/ai/v1")
    with pytest.raises(LLMConfigurationError):
        cloudflare_base_url("../evil")


def test_token_is_sent_only_as_the_bearer_header_and_never_in_the_body():
    w = Workers(ok({"role": "assistant", "content": "Hello!"}))
    chat(provider(w))
    req = w.requests[0]
    assert req.headers["authorization"] == f"Bearer {TOKEN}"
    assert TOKEN not in req.content.decode() and TOKEN not in str(req.url)
    config = LLMConfig.from_settings(cf_settings())
    assert TOKEN not in repr(config) and ACCOUNT not in repr(config)


@pytest.mark.parametrize(
    ("over", "needle"),
    [
        ({"cloudflare_account_id": None}, "CLOUDFLARE_ACCOUNT_ID"),
        ({"cloudflare_account_id": "not-hex"}, "CLOUDFLARE_ACCOUNT_ID"),
        ({"cloudflare_api_token": None}, "CLOUDFLARE_API_TOKEN"),
        ({"cloudflare_api_token": "   "}, "CLOUDFLARE_API_TOKEN"),
        ({"cloudflare_model": "bad model name!"}, "valid model name"),
    ],
)
def test_configuration_is_validated_without_echoing_secrets(over, needle):
    with pytest.raises(LLMConfigurationError) as info:
        LLMConfig.from_settings(cf_settings(**over))
    assert needle in info.value.message and TOKEN not in info.value.message


# --- 4-8: responses and tool calls ---------------------------------------------------------
def test_plain_answer():
    w = Workers(ok({"role": "assistant", "content": "Order ORD-1001 was delivered."}))
    result = chat(provider(w))
    assert result.message.content == "Order ORD-1001 was delivered."
    assert result.message.tool_calls == [] and result.provider == "cloudflare"
    assert result.model == MODEL and result.fallback_used is False and result.attempts == 1


def test_structured_tool_call_is_normalised_with_its_id():
    w = Workers(
        ok(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call("call_7", "get_order", order_number="ORD-1001")],
            },
            "tool_calls",
        )
    )
    msg = chat(provider(w)).message
    assert msg.tool_calls == [
        {
            "name": "get_order",
            "args": {"order_number": "ORD-1001"},
            "id": "call_7",
            "type": "tool_call",
        }
    ]
    assert msg.invalid_tool_calls == [] and msg.content == ""


def test_multiple_tool_calls_keep_order_and_ids():
    w = Workers(
        ok(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    tool_call("a1", "get_order", order_number="ORD-1001"),
                    tool_call("a2", "get_shipment", shipment_number="SHP-1003"),
                ],
            },
            "tool_calls",
        )
    )
    msg = chat(provider(w)).message
    assert [(c["id"], c["name"]) for c in msg.tool_calls] == [
        ("a1", "get_order"),
        ("a2", "get_shipment"),
    ]


def test_content_and_tool_calls_together_are_both_kept():
    w = Workers(
        ok(
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [tool_call("x1", "get_order", order_number="ORD-1")],
            },
            "tool_calls",
        )
    )
    msg = chat(provider(w)).message
    assert msg.content == "Let me check." and msg.tool_calls[0]["id"] == "x1"


def test_unparseable_tool_arguments_become_invalid_tool_calls_not_executions():
    bad = {
        "id": "b1",
        "type": "function",
        "function": {"name": "get_order", "arguments": "{not json"},
    }
    w = Workers(ok({"role": "assistant", "content": None, "tool_calls": [bad]}, "tool_calls"))
    msg = chat(provider(w)).message
    assert msg.tool_calls == [] and msg.invalid_tool_calls[0]["id"] == "b1"


def test_tool_round_trip_history_is_sent_in_openai_format_with_the_same_ids():
    w = Workers(ok({"role": "assistant", "content": "It is delivered."}))
    history = [
        SystemMessage("sys"),
        HumanMessage("Where is ORD-1001?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_order",
                    "args": {"order_number": "ORD-1001"},
                    "id": "call_9",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content='{"ok": true}', tool_call_id="call_9", name="get_order"),
    ]
    chat(provider(w), history)
    sent = w.body()["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool"]
    assert sent[2]["tool_calls"][0]["id"] == "call_9" and sent[3]["tool_call_id"] == "call_9"


def test_another_providers_history_is_normalised_for_this_request_only():
    gemini_turn = AIMessage(
        content=[{"type": "text", "text": "Checking.", "extras": {"signature": "c2ln"}}],
        tool_calls=[
            {
                "name": "get_order",
                "args": {"order_number": "ORD-1"},
                "id": "g1",
                "type": "tool_call",
            }
        ],
        additional_kwargs={"__gemini_function_call_thought_signatures__": {"g1": "c2ln"}},
        response_metadata={"model_provider": "google_genai"},
    )
    out = openai_compatible_messages([gemini_turn])
    assert out[0].content == "Checking." and out[0].tool_calls[0]["id"] == "g1"
    assert out[0].additional_kwargs == {} and gemini_turn.additional_kwargs  # original untouched
    w = Workers(ok({"role": "assistant", "content": "done"}))
    chat(provider(w), [HumanMessage("x"), gemini_turn, ToolMessage("{}", tool_call_id="g1")])
    assert "c2ln" not in w.requests[0].content.decode()


# --- 9-14: failures map to the safe taxonomy ---------------------------------------------------
@pytest.mark.parametrize(
    ("response", "code", "retried"),
    [
        (
            httpx.Response(429, json={"errors": [{"code": 3040, "message": f"rate {TOKEN}"}]}),
            "llm_rate_limited",
            True,
        ),
        (httpx.Response(503, json={"errors": [{"message": "upstream"}]}), "llm_unavailable", True),
        (httpx.Response(500, json={"errors": [{"message": "boom"}]}), "llm_unavailable", True),
        (
            httpx.Response(401, json={"errors": [{"message": "bad token"}]}),
            "llm_auth_failed",
            False,
        ),
        (
            httpx.Response(403, json={"errors": [{"message": "forbidden"}]}),
            "llm_auth_failed",
            False,
        ),
        (httpx.Response(402, json={"errors": [{"message": "quota"}]}), "llm_quota_exceeded", False),
        (httpx.Response(400, json={"errors": [{"message": "bad"}]}), "llm_request_rejected", False),
        (
            httpx.Response(404, json={"errors": [{"message": "no model"}]}),
            "llm_model_not_found",
            False,
        ),
    ],
)
def test_http_failures_are_classified(response, code, retried):
    w = Workers(response, response)
    with pytest.raises(LLMError) as info:
        chat(provider(w))
    assert info.value.code == code
    assert len(w.requests) == (2 if retried else 1)  # LLM_MAX_RETRIES=1, transient only
    assert TOKEN not in str(info.value) and TOKEN not in info.value.message
    assert "errors" not in info.value.message  # no raw provider body


def test_timeout_maps_to_llm_timeout_after_the_retry():
    w = Workers(httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"))
    with pytest.raises(LLMError) as info:
        chat(provider(w))
    assert info.value.code == "llm_timeout" and len(w.requests) == 2


def test_connection_error_maps_to_unavailable():
    w = Workers(httpx.ConnectError("down"), httpx.ConnectError("down"))
    with pytest.raises(LLMError) as info:
        chat(provider(w))
    assert info.value.code == "llm_unavailable"


def test_a_transient_failure_then_success_is_one_successful_call():
    w = Workers(httpx.Response(503, json={}), ok({"role": "assistant", "content": "ok"}))
    result = chat(provider(w))
    assert result.attempts == 2 and result.message.content == "ok"


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": True},
        {"id": "x", "object": "chat.completion", "created": 1, "model": MODEL, "choices": []},
        "not json at all",
    ],
)
def test_malformed_provider_responses_are_safe_errors(payload):
    body = payload if isinstance(payload, str) else json.dumps(payload)
    w = Workers(httpx.Response(200, content=body, headers={"content-type": "application/json"}))
    with pytest.raises(LLMError) as info:
        chat(provider(w))
    assert info.value.code.startswith("llm_") and TOKEN not in str(info.value)


def test_response_without_usable_content_is_returned_empty_for_the_graph_to_reject():
    """The provider layer does not invent an answer: an empty AIMessage reaches the graph,
    whose final-answer guard refuses it (agent_empty_answer / protocol error)."""
    w = Workers(ok({"role": "assistant", "content": ""}))
    msg = chat(provider(w)).message
    assert msg.content == "" and msg.tool_calls == []


# --- 15-17: tool schemas stay tenant-safe -----------------------------------------------------
def test_bound_tool_schemas_never_expose_tenant_runtime_or_sql():
    w = Workers(ok({"role": "assistant", "content": "hi"}))
    chat(provider(w), tools=[*TOOLS, policy_search_tool(), *action_tools()])
    tools = w.body()["tools"]
    names = {t["function"]["name"] for t in tools}
    assert "get_order" in names and "search_policy_knowledge" in names
    raw = json.dumps(tools).lower()
    for forbidden in ("tenant", "runtime", "sql", "select ", "database"):
        assert forbidden not in raw, forbidden
    for t in tools:
        params = t["function"]["parameters"]
        assert "tenant_id" not in params.get("properties", {})


def test_tool_allowlist_is_exactly_what_the_graph_binds():
    w = Workers(ok({"role": "assistant", "content": "hi"}))
    chat(provider(w), tools=TOOLS)
    assert sorted(t["function"]["name"] for t in w.body()["tools"]) == sorted(t.name for t in TOOLS)


# --- structured output (intent vehicle) ------------------------------------------------------
def test_structured_output_is_one_forced_tool_call_validated_by_pydantic():
    args = {"intent": "order_lookup", "entities": ["ORD-1001"], "confidence": 0.9}
    w = Workers(
        ok(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call("s1", "IntentAnalysis", **args)],
            },
            "tool_calls",
        )
    )
    result = provider(w).invoke_structured(
        IntentAnalysis, [HumanMessage("ORD-1001?")], operation="intent"
    )
    assert result.value == IntentAnalysis(**args) and result.provider == "cloudflare"
    body = w.body()
    assert (
        body["tool_choice"] == "required"
        and body["tools"][0]["function"]["name"] == "IntentAnalysis"
    )
    assert "response_format" not in body


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": '{"intent": "order_lookup"}'},  # JSON as text: refused
        {"role": "assistant", "content": None, "tool_calls": [tool_call("s1", "Other", x=1)]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("s1", "IntentAnalysis", intent="nope", confidence=2)],
        },
    ],
)
def test_structured_output_violations_are_output_errors(message):
    w = Workers(ok(message, "tool_calls"))
    with pytest.raises(LLMError) as info:
        provider(w).invoke_structured(IntentAnalysis, [HumanMessage("x")], operation="intent")
    assert info.value.code == "llm_output_invalid" and len(w.requests) == 1  # never retried


# --- 18: credentials never leak --------------------------------------------------------------
def test_credentials_never_appear_in_errors_logs_or_results(caplog):
    caplog.set_level("DEBUG")
    w = Workers(
        httpx.Response(429, json={"error": TOKEN}),
        httpx.Response(429, json={"error": TOKEN}),
    )
    with pytest.raises(LLMError) as info:
        chat(provider(w))
    text = caplog.text + str(info.value) + repr(info.value) + info.value.message
    assert TOKEN not in text
    assert ACCOUNT not in "".join(
        r.getMessage() for r in caplog.records if r.name.startswith("app")
    )
