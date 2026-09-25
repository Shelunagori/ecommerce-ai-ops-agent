"""Cloudflare Workers AI tool calls without a usable correlation id (production regression).

Workers AI's OpenAI-compatible endpoint can return a structured tool call whose ``id`` is
missing / null / empty. LangChain then yields ``tool_calls[i]["id"]`` = None or "" and the
graph (correctly, generically) refuses the batch with ``agent_protocol_error`` /
``tool_call_id``. The Cloudflare adapter assigns a server-side correlation id to exactly those
calls; nothing else about a tool call is ever changed.
"""

from __future__ import annotations

import json
import logging
import re

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.agent.llm.config import LLMConfig
from app.agent.llm.factory import build_provider, normalize_tool_call_ids
from app.agent.llm.provider import ChatModelProvider
from tests.llm.conftest import FAKE_KEY, settings
from tests.llm.test_cloudflare_provider import TOKEN, Workers, chat, ok, provider

ARGS = json.dumps({"shipment_number": "SHP-1003"})
GENERATED = re.compile(r"^cf_call_[0-9a-f]{32}$")


def raw_call(name="get_shipment", arguments=ARGS, **id_field):
    call = {"type": "function", "function": {"name": name, "arguments": arguments}}
    call.update(id_field)
    return call


def reply(*calls):
    return ok({"role": "assistant", "content": None, "tool_calls": list(calls)}, "tool_calls")


# --- 1/15: a provider id is preserved exactly -------------------------------------------------
def test_valid_provider_id_is_preserved_exactly():
    msg = chat(provider(Workers(reply(raw_call(id="chatcmpl-tool-8f2a"))))).message
    assert [c["id"] for c in msg.tool_calls] == ["chatcmpl-tool-8f2a"]


# --- 2/3: missing / null / empty ids get a server-side correlation id -------------------------
@pytest.mark.parametrize("id_field", [{}, {"id": None}, {"id": ""}, {"id": "   "}])
def test_missing_id_gets_a_generated_correlation_id(id_field):
    msg = chat(provider(Workers(reply(raw_call(**id_field))))).message
    [call] = msg.tool_calls
    assert GENERATED.fullmatch(call["id"]) and call["name"] == "get_shipment"
    assert call["args"] == {"shipment_number": "SHP-1003"}  # nothing else changed
    assert msg.invalid_tool_calls == []


# --- 4/7: unique, distinct, provider ids untouched in a mixed batch ---------------------------
def test_generated_ids_are_unique_within_and_across_calls():
    w = Workers(
        reply(
            raw_call(), raw_call(name="get_order", arguments=json.dumps({"order_number": "ORD-1"}))
        ),
        reply(raw_call()),
    )
    p = provider(w)
    first = chat(p).message
    second = chat(p).message
    ids = [c["id"] for c in first.tool_calls] + [c["id"] for c in second.tool_calls]
    assert all(GENERATED.fullmatch(i) for i in ids) and len(set(ids)) == 3


def test_only_the_missing_ids_are_generated_in_a_mixed_batch():
    msg = chat(provider(Workers(reply(raw_call(id="keep-me"), raw_call())))).message
    assert msg.tool_calls[0]["id"] == "keep-me"
    assert GENERATED.fullmatch(msg.tool_calls[1]["id"])


def test_duplicate_provider_ids_are_not_repaired():
    """Two calls with the SAME provider id are ambiguous: left as-is for the graph to refuse."""
    msg = chat(provider(Workers(reply(raw_call(id="dup"), raw_call(id="dup"))))).message
    assert [c["id"] for c in msg.tool_calls] == ["dup", "dup"]


# --- 9: malformed calls stay malformed --------------------------------------------------------
def test_malformed_arguments_stay_invalid_and_get_no_id():
    msg = chat(provider(Workers(reply(raw_call(arguments="{not json"))))).message
    assert msg.tool_calls == [] and len(msg.invalid_tool_calls) == 1
    assert not msg.invalid_tool_calls[0].get("id")


def test_the_normaliser_never_touches_names_arguments_or_invalid_calls():
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "no_such_tool", "args": {"sql": "DROP TABLE x"}, "id": None}],
        invalid_tool_calls=[{"name": "get_order", "args": "{bad", "id": None, "error": "x"}],
    )
    out, facts = normalize_tool_call_ids(ai)
    assert out.tool_calls[0]["name"] == "no_such_tool"
    assert out.tool_calls[0]["args"] == {"sql": "DROP TABLE x"}  # unchanged: the graph decides
    assert out.invalid_tool_calls == ai.invalid_tool_calls
    assert facts == {"tool_calls": 1, "tool_call_id_normalized": 1}
    assert ai.tool_calls[0]["id"] is None  # the provider's message itself is not mutated


def test_a_message_without_tool_calls_is_returned_unchanged():
    ai = AIMessage(content="Shipment SHP-1003 is delayed.")
    out, facts = normalize_tool_call_ids(ai)
    assert out is ai and facts == {}


def test_raw_tool_calls_in_additional_kwargs_follow_the_normalised_ids():
    msg = chat(provider(Workers(reply(raw_call())))).message
    raw = msg.additional_kwargs.get("tool_calls")
    assert raw is None or raw[0]["id"] == msg.tool_calls[0]["id"]


# --- observability: safe structural facts only ------------------------------------------------
def test_normalisation_is_logged_structurally_without_values(caplog):
    caplog.set_level(logging.INFO, logger="app.agent.llm")
    chat(provider(Workers(reply(raw_call()))))
    [record] = [r for r in caplog.records if r.getMessage() == "llm call"]
    assert (record.provider, record.tool_calls, record.tool_call_id_normalized) == (
        "cloudflare",
        1,
        1,
    )
    text = caplog.text
    for value in ("SHP-1003", TOKEN, "cf_call_"):
        assert value not in text


def test_no_normalisation_fields_when_ids_are_present(caplog):
    caplog.set_level(logging.INFO, logger="app.agent.llm")
    chat(provider(Workers(reply(raw_call(id="c1")))))
    [record] = [r for r in caplog.records if r.getMessage() == "llm call"]
    assert record.tool_calls == 1 and not hasattr(record, "tool_call_id_normalized")


# --- 13/14: Gemini and Ollama are unchanged ---------------------------------------------------
@pytest.mark.parametrize(
    "over",
    [{"llm_provider": "gemini", "gemini_api_key": FAKE_KEY}, {"llm_provider": "ollama"}],
)
def test_gemini_and_ollama_get_no_tool_call_id_normalisation(over):
    p = build_provider(LLMConfig.from_settings(settings(**over)))
    assert isinstance(p, ChatModelProvider) and p._normalize is None


def test_cloudflare_has_the_normaliser():
    config = LLMConfig.from_settings(
        settings(
            llm_provider="cloudflare",
            cloudflare_account_id="0123456789abcdef0123456789abcdef",
            cloudflare_api_token=TOKEN,
        )
    )
    p = build_provider(
        config, http_client=httpx.Client(transport=httpx.MockTransport(lambda r: None))
    )
    assert p._normalize is normalize_tool_call_ids
