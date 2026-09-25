"""Capability sequencing: a commerce tool and policy retrieval requested in ONE model turn.

Production (Cloudflare ``@cf/zai-org/glm-4.7-flash``): "Where is SHP-1003, and what
compensation applies if it is delayed?" failed with ``agent_protocol_error`` /
``mixed_capability_batch`` - the model asked for get_shipment AND search_policy_knowledge in
the same response. The guard stays: nothing of that batch executes. The graph now makes ONE
corrective model call (same provider, same history plus an application note) and continues
sequentially; a second mixed batch fails closed with the same error.
"""

from __future__ import annotations

import json
import logging

import pytest

from app.actions.capability import PROPOSE_CANCEL_ORDER
from app.agent.assistant import AssistantLimits
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.profile import AGENT_PROFILE
from app.agent.llm.provider import ChatModelProvider, ProviderInfo, RetryPolicy
from app.agent.prompts import graph_agent, graph_demo, graph_rag
from app.api.agent_runtime import AgentRuntime
from tests.assistant.fakes import ScriptedChatModel, ai_text, ai_tools, call
from tests.db.test_agent_stream import finished
from tests.db.test_commerce_grounding_routing import GeminiRig
from tests.db.test_provider_fallback_graph import Workers, cf_text, cf_tools
from tests.db.test_provider_fallback_graph import tool_runs as tool_runs  # noqa: F401 (fixture)
from tests.llm.test_cloudflare_provider import tool_call

GLM = "@cf/zai-org/glm-4.7-flash"
V2 = "policy://delayed-shipment-compensation/v2#chunk-2"
QUESTION = "Where is SHP-1003, and what compensation applies if it is delayed?"


@pytest.fixture
def rig(committed):
    return GeminiRig(committed)


def shipment(cid="s1", **extra):
    return tool_call(cid, "get_shipment", shipment_number="SHP-1003", **extra)


def search(cid="p1"):
    return tool_call(cid, "search_policy_knowledge", query="delayed shipment compensation")


MIXED = cf_tools(shipment("m1"), search("m2"))


def steps(trace):
    return [(t["kind"], t["status"]) for t in trace]


def sequencing(trace):
    from app.agent.trace import CAPABILITY_SEQUENCING_LABEL

    return [t for t in trace if t["label"] == CAPABILITY_SEQUENCING_LABEL]


CORRECTED = [
    ("request", "completed"),
    ("model", "completed"),  # the rejected mixed batch (nothing executed)
    ("model", "rejected"),  # capability sequencing: rejected, NOT failed
    ("model", "completed"),  # corrected: get_shipment only
    ("commerce_tool", "completed"),
    ("model", "completed"),  # policy retrieval in a later turn
    ("retrieval", "completed"),
    ("model", "completed"),  # final answer
    ("grounding", "completed"),
    ("response", "completed"),
]


# --- 1-5, 8, 9, 14: the GLM mixed trajectory succeeds through ONE correction ----------------
def test_glm_mixed_batch_is_corrected_then_runs_sequentially(rig, tenant_b, tool_runs):
    from app.agent.graph.nodes import CAPABILITY_CORRECTION_NOTE

    rig.install(
        Workers(MIXED, cf_tools(shipment()), cf_tools(search()), cf_text(f"Delayed [{V2}].")),
        cloudflare_model=GLM,
    )
    body = rig.post(tenant_b, QUESTION).json()
    assert body["answer"] == f"Delayed [{V2}]." and [c["citation"] for c in body["citations"]] == [
        V2
    ]
    trace = body["execution_trace"]
    assert steps(trace) == CORRECTED
    assert not [t for t in trace if t["status"] == "failed"]
    assert tool_runs["names"] == ["get_shipment"]  # exactly once, from the corrected turn
    assert len(rig.retriever.calls) == 1 and body["model_calls"] == 4
    assert [t["tool"] for t in body["tool_calls"]] == ["get_shipment"]

    # the corrective request: same history + the note; the rejected batch is NOT sent
    corrective = rig.workers.body(1)["messages"]
    assert corrective[-1] == {"role": "user", "content": CAPABILITY_CORRECTION_NOTE}
    assert corrective[:-1] == rig.workers.body(0)["messages"]
    sent_ids = {c["id"] for m in rig.workers.body(3)["messages"] for c in m.get("tool_calls") or []}
    assert sent_ids == {"s1", "p1"}  # m1 / m2 never reach any later request

    # durable history: neither the rejected batch nor the note is kept
    [latest] = list(rig.saver.list(None, limit=1))
    values = latest.checkpoint["channel_values"]
    kept = values["messages"]
    assert CAPABILITY_CORRECTION_NOTE not in [m.content for m in kept]
    assert {c["id"] for m in kept for c in getattr(m, "tool_calls", [])} == {"s1", "p1"}
    assert "m1" not in values["seen_tool_call_ids"] and "m2" not in values["seen_tool_call_ids"]
    assert values["invalid_tool_calls"] == []


# --- 6, 7, Test C: a second mixed batch fails closed; ONE correction per run ------------------
def test_a_second_mixed_batch_fails_closed(rig, tenant_b, tool_runs):
    rig.install(Workers(MIXED, cf_tools(shipment("x1"), search("x2")), cf_text("never")))
    r = rig.post(tenant_b, QUESTION)
    assert (r.status_code, r.json()["error"]["code"]) == (502, "agent_protocol_error")
    assert tool_runs["n"] == 0 and rig.retriever.calls == []
    assert len(rig.workers.requests) == 2  # original + ONE corrective call, never a third
    assert len(rig.workers.script) == 1  # the scripted third answer was never requested


def test_only_one_correction_per_run_even_after_progress(rig, tenant_b, tool_runs):
    rig.install(
        Workers(MIXED, cf_tools(shipment()), cf_tools(shipment("y1"), search("y2")), cf_text("x"))
    )
    r = rig.post(tenant_b, QUESTION)
    assert r.json()["error"]["code"] == "agent_protocol_error"
    assert tool_runs["names"] == ["get_shipment"] and rig.retriever.calls == []
    assert len(rig.workers.requests) == 3


def test_no_correction_without_a_model_round_left(committed, tenant_b, tool_runs):
    rig = GeminiRig(committed)
    rig.gemini = ScriptedChatModel(
        [
            ai_tools(call("get_shipment", "m1", shipment_number="SHP-1003"), _gsearch("m2")),
            ai_text("never"),
        ]
    )
    provider = ChatModelProvider(
        rig.gemini,  # type: ignore[arg-type]
        ProviderInfo(provider="gemini", model="gemini-3.8-flash"),
        RetryPolicy(max_retries=0),
        sleep=lambda _s: None,
    )
    rig.assistant = CommerceGraphAssistant(
        provider,
        tools=rig.tools,
        checkpointer=rig.saver,
        retriever=rig.retriever,
        profile=AGENT_PROFILE,
        actions=rig.actions,
        limits=AssistantLimits(max_model_rounds=1),
    )
    rig.app.state.agent_runtime = AgentRuntime(rig.assistant, rig.actions)
    r = rig.post(tenant_b, QUESTION)
    assert r.json()["error"]["code"] == "agent_protocol_error"
    assert len(rig.gemini.invocations) == 1 and tool_runs["n"] == 0


# --- 10, 11, 13: no correction for valid sequential requests ---------------------------------
def test_commerce_only_request_gets_no_correction(rig, tenant_a, tool_runs):
    rig.install(Workers(cf_tools(shipment()), cf_text("Shipment SHP-1003 is delayed.")))
    body = rig.post(tenant_a, "Where is SHP-1003?").json()
    assert body["model_calls"] == 2 and not sequencing(body["execution_trace"])
    assert tool_runs["names"] == ["get_shipment"]


def test_policy_only_request_gets_no_correction(rig, tenant_b):
    rig.install(Workers(cf_tools(search()), cf_text(f"15% store credit [{V2}].")))
    body = rig.post(tenant_b, "What compensation applies to a delayed shipment?").json()
    assert body["model_calls"] == 2 and not sequencing(body["execution_trace"])
    assert [c["citation"] for c in body["citations"]] == [V2]


def test_cloudflare_llama_sequential_trajectory_is_unchanged(rig, tenant_b, tool_runs):
    rig.install(Workers(cf_tools(shipment()), cf_tools(search()), cf_text(f"Delayed [{V2}].")))
    body = rig.post(tenant_b, QUESTION).json()
    assert body["model_calls"] == 3 and not sequencing(body["execution_trace"])
    assert [k for k, _ in steps(body["execution_trace"])] == [
        "request",
        "model",
        "commerce_tool",
        "model",
        "retrieval",
        "model",
        "grounding",
        "response",
    ]


# --- 12: Gemini ---------------------------------------------------------------------------------
def _gsearch(cid):
    return call("search_policy_knowledge", cid, query="delayed shipment compensation")


def test_gemini_sequential_mixed_trajectory_is_unchanged(rig, tenant_b, tool_runs):
    rig.gemini_only(
        ai_tools(call("get_shipment", "g1", shipment_number="SHP-1003")),
        ai_tools(_gsearch("g2")),
        ai_text(f"Delayed [{V2}]."),
    )
    body = rig.post(tenant_b, QUESTION).json()
    assert body["model_calls"] == 3 and not sequencing(body["execution_trace"])
    assert tool_runs["names"] == ["get_shipment"]


def test_gemini_mixed_batch_gets_the_same_single_correction(rig, tenant_b, tool_runs):
    rig.gemini_only(
        ai_tools(call("get_shipment", "g1", shipment_number="SHP-1003"), _gsearch("g2")),
        ai_tools(call("get_shipment", "g3", shipment_number="SHP-1003")),
        ai_tools(_gsearch("g4")),
        ai_text(f"Delayed [{V2}]."),
    )
    body = rig.post(tenant_b, QUESTION).json()
    assert steps(body["execution_trace"]) == CORRECTED and tool_runs["names"] == ["get_shipment"]
    assert len(rig.gemini.invocations) == 4


# --- scope guards: action mixes stay fail-closed (HITL untouched) -----------------------------
def test_commerce_plus_action_mix_is_not_corrected(rig, tenant_b, tool_runs):
    rig.gemini_only(
        ai_tools(
            call("get_order", "a1", order_number="ORD-1004"),
            call(PROPOSE_CANCEL_ORDER, "a2", order_number="ORD-1004", reason="late"),
        ),
        ai_text("never"),
    )
    r = rig.post(tenant_b, "Cancel ORD-1004")
    assert r.json()["error"]["code"] == "agent_protocol_error"
    assert len(rig.gemini.invocations) == 1 and tool_runs["n"] == 0


def test_policy_plus_action_mix_is_not_corrected(rig, tenant_b, tool_runs):
    rig.gemini_only(
        ai_tools(
            _gsearch("b1"),
            call(PROPOSE_CANCEL_ORDER, "b2", order_number="ORD-1004", reason="late"),
        ),
        ai_text("never"),
    )
    r = rig.post(tenant_b, "Cancel ORD-1004 per policy")
    assert r.json()["error"]["code"] == "agent_protocol_error"
    assert len(rig.gemini.invocations) == 1 and rig.retriever.calls == []


# --- 15, 16, 17: tenant scope, allowlist, no SQL -----------------------------------------------
def test_tenant_and_allowlist_are_unchanged_by_the_correction(rig, tenant_a, tenant_b, tool_runs):
    rig.install(
        Workers(
            MIXED,
            cf_tools(shipment(tenant_id=str(tenant_a.tenant_id))),
            cf_tools(search()),
            cf_text(f"Delayed [{V2}]."),
        ),
        cloudflare_model=GLM,
    )
    body = rig.post(tenant_b, QUESTION).json()
    assert body["tool_calls"][0]["rejected_argument_names"] == ["tenant_id"]
    assert [c.tenant_id for c in rig.retriever.calls] == [tenant_b.tenant_id]
    assert str(tenant_a.tenant_id) not in json.dumps(body)
    first, corrective = rig.workers.body(0), rig.workers.body(1)
    assert corrective["tools"] == first["tools"]  # identical allowlist on the corrective call
    names = {t["function"]["name"] for t in corrective["tools"]}
    assert not {n for n in names if "sql" in n.lower() or "query" in n.lower()}


# --- 18, 19: streaming == JSON; the live trace records the correction safely ------------------
def test_stream_and_json_agree_and_the_trace_is_safe(rig, tenant_b):
    script = (MIXED, cf_tools(shipment()), cf_tools(search()), cf_text(f"Delayed [{V2}]."))
    rig.install(Workers(*script), cloudflare_model=GLM)
    plain = rig.post(tenant_b, QUESTION, thread="p").json()
    rig.install(Workers(*script), cloudflare_model=GLM)
    events = rig.stream(tenant_b, QUESTION, thread="s")
    assert events[-1]["type"] == "run_completed"
    assert events[-1]["response"]["answer"] == plain["answer"]
    done = finished(events)
    assert [(e["kind"], e["status"]) for e in done] == steps(plain["execution_trace"])
    [live] = [e for e in done if e["label"] == "Capability sequencing"]
    [final] = sequencing(plain["execution_trace"])
    assert live["status"] == "rejected" and live["type"] == "step_completed"
    assert (live["detail"], live["metadata"]) == (final["detail"], final["metadata"])
    assert final["metadata"] == {"protocol_issue": "mixed_capability_batch"}
    text = json.dumps([e for e in events if e["type"] != "run_completed"])
    for leaked in ("SHP-1003", "delayed shipment compensation", "m1", "m2"):
        assert leaked not in text


def test_stream_reports_a_second_violation_as_a_failed_run(rig, tenant_b):
    rig.install(Workers(MIXED, cf_tools(shipment("x1"), search("x2"))))
    events = rig.stream(tenant_b, QUESTION)
    assert events[-1]["type"] == "run_failed"
    assert events[-1]["error"]["code"] == "agent_protocol_error"
    failed = [e for e in finished(events) if e["status"] == "failed"]
    assert [e["label"] for e in failed] == ["Agent orchestration"]
    assert failed[0]["detail"].startswith("Model response rejected")


# --- 20: the correction never duplicates a completed tool --------------------------------------
def test_correction_after_a_completed_tool_does_not_replay_it(rig, tenant_b, tool_runs):
    rig.install(
        Workers(
            cf_tools(shipment()),
            cf_tools(shipment("m1"), search("m2")),  # mixed AFTER get_shipment already ran
            cf_tools(search()),
            cf_text(f"Delayed [{V2}]."),
        ),
        cloudflare_model=GLM,
    )
    body = rig.post(tenant_b, QUESTION).json()
    assert body["answer"] == f"Delayed [{V2}]." and tool_runs["names"] == ["get_shipment"]
    assert len(sequencing(body["execution_trace"])) == 1 and len(rig.retriever.calls) == 1


# --- observability and prompt ------------------------------------------------------------------
def test_run_log_counts_the_correction(rig, tenant_b, caplog):
    caplog.set_level(logging.INFO, logger="app.agent.graph")
    rig.install(
        Workers(MIXED, cf_tools(shipment()), cf_tools(search()), cf_text(f"Delayed [{V2}]."))
    )
    rig.post(tenant_b, QUESTION)
    [record] = [r for r in caplog.records if r.getMessage() == "graph assistant run"]
    assert (record.capability_corrections, record.invalid_tool_calls) == (1, 0)
    assert record.outcome == "ok"


def test_prompts_state_the_one_capability_per_step_rule():
    assert graph_agent.PROMPT_VERSION == "commerce-assistant-v5"
    assert graph_demo.PROMPT_VERSION == "commerce-assistant-v5-public-demo"
    for prompt in (graph_agent.SYSTEM_PROMPT, graph_demo.SYSTEM_PROMPT):
        assert graph_agent.CAPABILITY_RULES.strip() in prompt
    assert graph_agent.CAPABILITY_RULES.strip() not in graph_rag.SYSTEM_PROMPT  # v2 unchanged
    assert graph_rag.PROMPT_VERSION == "commerce-assistant-v2"
