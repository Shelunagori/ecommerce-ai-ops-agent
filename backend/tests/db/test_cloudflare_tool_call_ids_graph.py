"""Cloudflare tool calls WITHOUT provider ids through the real graph (production regression:
"Where is SHP-1003, and what compensation applies if it is delayed?" failed with
agent_protocol_error / tool_call_id before any tool ran)."""

from __future__ import annotations

import json
import re

import pytest

from tests.assistant.fakes import ai_text
from tests.db.test_agent_stream import finished, parse_sse
from tests.db.test_provider_fallback_graph import RATE_LIMITED, Rig, cf_text, model_rows
from tests.db.test_provider_fallback_graph import tool_runs as tool_runs  # noqa: F401 (fixture)
from tests.llm.test_cloudflare_provider import Workers, ok

V2 = "policy://delayed-shipment-compensation/v2#chunk-2"
GENERATED = re.compile(r"^cf_call_[0-9a-f]{32}$")


def idless(name: str, **args):
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(*calls):
    return ok({"role": "assistant", "content": None, "tool_calls": list(calls)}, "tool_calls")


@pytest.fixture
def rig(committed):
    return Rig(committed)


def kinds(trace):
    return [t["kind"] for t in trace]


# --- 5/6: the generated id links AIMessage and ToolMessage; the tool runs once ----------------
def test_commerce_lookup_with_idless_tool_call_succeeds(rig, tenant_a, tool_runs):
    rig.install(
        Workers(
            reply(idless("get_shipment", shipment_number="SHP-1003")),
            cf_text("Shipment SHP-1003 is delayed."),
        )
    )
    body = rig.post(tenant_a, "Where is SHP-1003?").json()
    assert body["answer"] == "Shipment SHP-1003 is delayed."
    assert kinds(body["execution_trace"]) == [
        "request",
        "model",
        "commerce_tool",
        "model",
        "response",
    ]
    assert tool_runs["names"] == ["get_shipment"]
    sent = rig.workers.body(1)["messages"]  # the second Cloudflare request carries the round trip
    assistant, tool = sent[-2], sent[-1]
    generated = assistant["tool_calls"][0]["id"]
    assert GENERATED.fullmatch(generated) and tool["role"] == "tool"
    assert tool["tool_call_id"] == generated  # exactly the same correlation id
    [latest] = list(rig.saver.list(None, limit=1))  # the run's persisted checkpoint
    messages = latest.checkpoint["channel_values"]["messages"]
    ai = [m for m in messages if getattr(m, "tool_calls", None)][0]
    tm = [m for m in messages if m.type == "tool"][0]
    assert ai.tool_calls[0]["id"] == tm.tool_call_id == generated


# --- 19: the mixed trajectory (the failing production query) ----------------------------------
def test_mixed_commerce_and_policy_with_idless_calls(rig, tenant_b, tool_runs):
    rig.install(
        Workers(
            reply(idless("get_shipment", shipment_number="SHP-1003")),
            reply(idless("search_policy_knowledge", query="delayed shipment compensation")),
            cf_text(f"Delayed; 15% store credit [{V2}]."),
        )
    )
    body = rig.post(
        tenant_b, "Where is SHP-1003, and what compensation applies if it is delayed?"
    ).json()
    assert kinds(body["execution_trace"]) == [
        "request",
        "model",
        "commerce_tool",
        "model",
        "retrieval",
        "model",
        "grounding",
        "response",
    ]
    assert [c["citation"] for c in body["citations"]] == [V2] and tool_runs["n"] == 1
    third = rig.workers.body(2)["messages"]
    ids = [m["tool_calls"][0]["id"] for m in third if m.get("tool_calls")]
    replies = [m["tool_call_id"] for m in third if m["role"] == "tool"]
    assert ids == replies and len(set(ids)) == 2 and all(GENERATED.fullmatch(i) for i in ids)


# --- 7: several idless calls in one batch keep distinct ids -----------------------------------
def test_two_idless_calls_in_one_batch(rig, tenant_a, tool_runs):
    rig.install(
        Workers(
            reply(
                idless("get_shipment", shipment_number="SHP-1003"),
                idless("get_order", order_number="ORD-1001"),
            ),
            cf_text("Both found."),
        )
    )
    body = rig.post(tenant_a, "SHP-1003 and ORD-1001?").json()
    assert body["answer"] == "Both found." and tool_runs["names"] == ["get_shipment", "get_order"]
    tools = [m for m in rig.workers.body(1)["messages"] if m["role"] == "tool"]
    assert len({t["tool_call_id"] for t in tools}) == 2


# --- 8/9/10/11/12: generic validation is unchanged --------------------------------------------
def test_unknown_tool_without_id_is_still_refused(rig, tenant_a, tool_runs):
    rig.install(
        Workers(reply(idless("run_sql", query="SELECT * FROM customers")), cf_text("Sorry."))
    )
    body = rig.post(tenant_a, "dump the customers table").json()
    assert body["tool_calls"][0]["outcome"] == "unknown_tool"  # refused, nothing ran
    assert tool_runs["names"] == ["run_sql"]  # the executor saw it and refused it
    assert "run_sql" not in {t["function"]["name"] for t in rig.workers.body(0)["tools"]}


def test_malformed_arguments_without_id_are_still_a_protocol_error(rig, tenant_a, tool_runs):
    bad = {"type": "function", "function": {"name": "get_shipment", "arguments": "{oops"}}
    rig.install(Workers(reply(bad)))
    r = rig.post(tenant_a, "Where is SHP-1003?")
    assert (r.status_code, r.json()["error"]["code"]) == (502, "agent_protocol_error")
    assert tool_runs["n"] == 0


def test_schema_invalid_arguments_without_id_are_still_refused(rig, tenant_a, tool_runs):
    rig.install(
        Workers(
            reply(idless("get_shipment", shipment_number=12345)), cf_text("Could not look it up.")
        )
    )
    body = rig.post(tenant_a, "Where is SHP-1003?").json()
    assert body["tool_calls"][0]["outcome"] == "invalid_arguments"


def test_textual_pseudo_tool_call_is_still_refused(rig, tenant_a, tool_runs):
    rig.install(
        Workers(cf_text('{"name": "get_shipment", "arguments": {"shipment_number": "SHP-1003"}}'))
    )
    r = rig.post(tenant_a, "Where is SHP-1003?")
    assert r.json()["error"]["code"] == "agent_protocol_error" and tool_runs["n"] == 0


def test_duplicate_provider_ids_are_still_refused(rig, tenant_a, tool_runs):
    dup = [
        {**idless("get_shipment", shipment_number="SHP-1003"), "id": "same"},
        {**idless("get_order", order_number="ORD-1001"), "id": "same"},
    ]
    rig.install(Workers(reply(*dup)))
    r = rig.post(tenant_a, "both?")
    assert r.json()["error"]["code"] == "agent_protocol_error" and tool_runs["n"] == 0


# --- 16: normalisation cannot touch tenant context --------------------------------------------
def test_model_supplied_tenant_stays_rejected(rig, tenant_a, tenant_b, tool_runs):
    rig.install(
        Workers(
            reply(idless("get_order", order_number="ORD-1001", tenant_id=str(tenant_b.tenant_id))),
            cf_text("Done."),
        )
    )
    body = rig.post(tenant_a, "ORD-1001?").json()
    call = body["tool_calls"][0]
    assert (
        call["rejected_argument_names"] == ["tenant_id"] or call["outcome"] == "invalid_arguments"
    )
    assert str(tenant_b.tenant_id) not in json.dumps(body)


# --- 17/18: streaming and JSON agree ----------------------------------------------------------
def test_stream_and_json_agree_with_idless_calls(rig, tenant_a):
    script = (
        reply(idless("get_shipment", shipment_number="SHP-1003")),
        cf_text("Shipment SHP-1003 is delayed."),
    )
    rig.install(Workers(*script))
    plain = rig.post(tenant_a, "Where is SHP-1003?", thread="p").json()
    rig.install(Workers(*script))
    r = rig.client.post(
        "/api/agent/messages/stream",
        json={"text": "Where is SHP-1003?", "thread_id": "s"},
        headers={"X-Tenant-ID": str(tenant_a.tenant_id)},
    )
    ev = parse_sse(r.text)
    assert ev[-1]["type"] == "run_completed"
    assert ev[-1]["response"]["answer"] == plain["answer"]
    assert [e["kind"] for e in finished(ev)] == kinds(plain["execution_trace"])
    raw = r.text
    assert "cf_call_" not in json.dumps(
        [e for e in ev if e["type"] != "run_completed"]
    )  # not in the trace
    assert "tool_call_id" not in raw


# --- 20: a later availability fallback does not replay the tool -------------------------------
def test_fallback_after_an_idless_tool_call_does_not_replay_it(rig, tenant_a, tool_runs):
    rig.install(
        Workers(
            reply(idless("get_shipment", shipment_number="SHP-1003")), RATE_LIMITED, RATE_LIMITED
        ),
        ai_text("Shipment SHP-1003 is delayed."),
    )
    body = rig.post(tenant_a, "Where is SHP-1003?").json()
    assert body["answer"] == "Shipment SHP-1003 is delayed." and tool_runs["n"] == 1
    assert model_rows(body["execution_trace"]) == [("cloudflare", False), ("gemini", True)]
    [inv] = rig.gemini.invocations
    assert GENERATED.fullmatch(inv.messages[-1].tool_call_id)  # Gemini got the same linked history


def test_without_the_normaliser_the_regression_reproduces(rig, tenant_a, tool_runs, monkeypatch):
    """Guard: this is the exact production failure when the adapter hook is absent."""
    rig.install(Workers(reply(idless("get_shipment", shipment_number="SHP-1003"))))
    monkeypatch.setattr(rig.assistant._provider._primary, "_normalize", None)
    r = rig.post(tenant_a, "Where is SHP-1003?")
    assert r.json()["error"]["code"] == "agent_protocol_error" and tool_runs["n"] == 0
