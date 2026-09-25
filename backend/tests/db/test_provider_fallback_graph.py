"""Cloudflare Workers AI as the primary chat provider with Gemini fallback, through the REAL
graph on real PostgreSQL: real tools, actions, checkpoints, grounding and the live trace.

Cloudflare is the real OpenAI-compatible client behind an ``httpx.MockTransport`` (scripted
Workers AI responses); Gemini is a scripted chat model behind the same ``ChatModelProvider``.
Assertions are about trajectories (what ran, how often, who answered), not prose.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
from sqlalchemy import func, select

from app.actions.capability import PROPOSE_CANCEL_ORDER
from app.agent.assistant import AssistantError
from app.agent.assistant.executor import ToolExecutor
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.profile import AGENT_PROFILE, PUBLIC_DEMO_PROFILE
from app.agent.llm.config import LLMConfig
from app.agent.llm.factory import build_provider
from app.agent.llm.provider import (
    ChatModelProvider,
    FallbackProvider,
    ProviderInfo,
    RetryPolicy,
)
from app.api.agent_runtime import AgentRuntime
from app.models import ActionRequest, AuditEvent
from tests.assistant.fakes import ScriptedChatModel, ai_text, ai_tools, call
from tests.db.test_agent_api import V2, Harness, order_status
from tests.db.test_agent_stream import finished, parse_sse
from tests.llm.test_cloudflare_provider import Workers, cf_settings, ok, tool_call
from tests.rag.fakes import result as retrieval_result

RATE_LIMITED = httpx.Response(429, json={"errors": [{"code": 3040, "message": "capacity"}]})


def cf_tools(*calls: dict) -> httpx.Response:
    return ok({"role": "assistant", "content": None, "tool_calls": list(calls)}, "tool_calls")


def cf_text(text: str) -> httpx.Response:
    return ok({"role": "assistant", "content": text})


class Rig(Harness):
    """Agent API harness whose model is Cloudflare (mocked HTTP) with a Gemini fallback."""

    def install(self, workers: Workers, *gemini_steps, profile=AGENT_PROFILE, fallback=True):
        cf = build_provider(
            LLMConfig.from_settings(cf_settings()),
            http_client=httpx.Client(transport=httpx.MockTransport(workers)),
        )
        cf._sleep = lambda _s: None
        self.workers = workers
        self.gemini = ScriptedChatModel(list(gemini_steps))
        gm = ChatModelProvider(
            self.gemini,  # type: ignore[arg-type]
            ProviderInfo(provider="gemini", model="gemini-3.8-flash"),
            RetryPolicy(max_retries=1),
            sleep=lambda _s: None,
        )
        provider = FallbackProvider(cf, gm) if fallback else cf
        common = {
            "tools": self.tools,
            "checkpointer": self.saver,
            "retriever": self.retriever,
            "run_recorder": None,
        }
        if profile is AGENT_PROFILE:
            self.assistant = CommerceGraphAssistant(
                provider, profile=AGENT_PROFILE, actions=self.actions, **common
            )
        else:
            self.assistant = CommerceGraphAssistant(provider, profile=profile, **common)
        self.app.state.agent_runtime = AgentRuntime(self.assistant, self.actions)
        return self

    def stream(self, tenant, text, thread="thread-1"):
        r = self.client.post(
            "/api/agent/messages/stream",
            json={"text": text, "thread_id": thread},
            headers={"X-Tenant-ID": str(tenant.tenant_id)},
        )
        assert r.status_code == 200, r.text
        return parse_sse(r.text)


@pytest.fixture
def rig(committed):
    return Rig(committed)


@pytest.fixture
def tool_runs(monkeypatch):
    """Counts every real commerce tool execution (to prove nothing is replayed)."""
    count = {"n": 0, "names": []}
    real = ToolExecutor.execute

    def counting(self, call_, context, round_no):
        count["n"] += 1
        count["names"].append(call_.get("name"))
        return real(self, call_, context, round_no)

    monkeypatch.setattr(ToolExecutor, "execute", counting)
    return count


def model_rows(trace):
    return [
        (t["metadata"]["provider"], t["metadata"].get("fallback_used", False))
        for t in trace
        if t["kind"] == "model"
    ]


def kinds(trace):
    return [t["kind"] for t in trace]


# --- agent integration: Cloudflare as primary --------------------------------------------------
def test_commerce_lookup_with_cloudflare(rig, tenant_a, tool_runs):
    rig.install(
        Workers(
            cf_tools(tool_call("c1", "get_order", order_number="ORD-1001")),
            cf_text("ORD-1001 was delivered."),
        ),
    )
    body = rig.post(tenant_a, "Show me order ORD-1001").json()
    assert kinds(body["execution_trace"]) == [
        "request",
        "model",
        "commerce_tool",
        "model",
        "response",
    ]
    assert model_rows(body["execution_trace"]) == [("cloudflare", False), ("cloudflare", False)]
    assert body["tool_calls"][0]["outcome"] == "success" and tool_runs["names"] == ["get_order"]
    # round 2 sent the tool result back with the SAME tool-call id
    sent = rig.workers.body(1)["messages"]
    assert sent[-2]["tool_calls"][0]["id"] == "c1" and sent[-1]["tool_call_id"] == "c1"
    assert rig.gemini.invocations == []


def test_policy_rag_with_cloudflare_keeps_retrieval_and_grounding(rig, tenant_b):
    rig.install(
        Workers(
            cf_tools(
                tool_call("r1", "search_policy_knowledge", query="delayed shipment compensation")
            ),
            cf_text(f"15% store credit [{V2}]."),
        )
    )
    body = rig.post(tenant_b, "What compensation applies?").json()
    assert kinds(body["execution_trace"]) == [
        "request",
        "model",
        "retrieval",
        "model",
        "grounding",
        "response",
    ]
    assert [c["citation"] for c in body["citations"]] == [V2]
    assert rig.retriever.calls[0].tenant_id == tenant_b.tenant_id  # trusted tenant, unchanged
    retrieval = next(t for t in body["execution_trace"] if t["kind"] == "retrieval")
    assert retrieval["metadata"]["retriever"] == "semantic-pgvector-v1"  # embedding path untouched


def test_mixed_commerce_and_policy_with_cloudflare(rig, tenant_b, tool_runs):
    rig.install(
        Workers(
            cf_tools(tool_call("m1", "get_shipment", shipment_number="SHP-1003")),
            cf_tools(
                tool_call("m2", "search_policy_knowledge", query="delayed shipment compensation")
            ),
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
    assert model_rows(body["execution_trace"]) == [("cloudflare", False)] * 3
    assert tool_runs["names"] == ["get_shipment"]


def test_no_result_rag_with_cloudflare(rig, tenant_b):
    rig.retriever.script.append(retrieval_result())  # no eligible policy section
    rig.install(
        Workers(
            cf_tools(tool_call("n1", "search_policy_knowledge", query="warranty on spaceships")),
            cf_text("I could not find a policy that covers this."),
        )
    )
    body = rig.post(tenant_b, "Warranty on spaceships?").json()
    assert body["retrievals"][0]["outcome"] == "no_results" and body["citations"] == []
    assert "grounding" in kinds(body["execution_trace"])


def test_public_demo_mutation_request_with_cloudflare_writes_nothing(rig, tenant_b, committed):
    rig.install(
        Workers(
            cf_tools(tool_call("p1", PROPOSE_CANCEL_ORDER, order_number="ORD-1004", reason="x")),
            cf_text("The public demo is read-only."),
        ),
        profile=PUBLIC_DEMO_PROFILE,
    )
    res = rig.assistant.run("Cancel ORD-1004", AgentContext(tenant_b.tenant_id), thread_id="demo")
    names = {t["function"]["name"] for t in rig.workers.body(0)["tools"]}
    assert PROPOSE_CANCEL_ORDER not in names  # no action capability is even offered
    assert res.action is None and res.tool_calls[0].outcome == "unknown_tool"
    with committed() as s:
        assert s.scalar(select(func.count()).select_from(ActionRequest)) == 0


def test_hitl_proposal_with_cloudflare_then_single_execution(rig, tenant_a, committed):
    rig.install(
        Workers(
            cf_tools(
                tool_call(
                    "h1", PROPOSE_CANCEL_ORDER, order_number="ORD-1004", reason="Customer request"
                )
            )
        )
    )
    body = rig.post(tenant_a, "Cancel ORD-1004").json()
    action = body["action"]
    assert action["status"] == "pending_approval" and order_status(rig, tenant_a) == "processing"
    decided = rig.decide(tenant_a, action["id"], arguments_hash=action["arguments_hash"]).json()
    assert decided["action"]["status"] == "succeeded" and order_status(rig, tenant_a) == "cancelled"
    assert len(rig.workers.requests) == 1  # resume executes deterministically: no model call
    with committed() as s:
        assert (
            s.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(AuditEvent.event_type == "action_succeeded")
            )
            == 1
        )


# --- H: fallback on model call #2, after a commerce tool -----------------------------------------
def test_fallback_on_call_two_does_not_replay_the_commerce_tool(rig, tenant_a, tool_runs):
    rig.install(
        Workers(
            cf_tools(tool_call("f1", "get_shipment", shipment_number="SHP-1003")),
            RATE_LIMITED,
            RATE_LIMITED,
        ),
        ai_text("Shipment SHP-1003 is delayed."),
    )
    body = rig.post(tenant_a, "Where is SHP-1003?").json()
    assert body["answer"] == "Shipment SHP-1003 is delayed."
    assert tool_runs["n"] == 1 and len(body["tool_calls"]) == 1  # executed exactly once
    assert model_rows(body["execution_trace"]) == [("cloudflare", False), ("gemini", True)]
    # Gemini answered call #2 from the exact same history, including the tool result
    [inv] = rig.gemini.invocations
    assert [type(m).__name__ for m in inv.messages][-2:] == ["AIMessage", "ToolMessage"]
    assert inv.messages[-1].tool_call_id == "f1"
    assert len(rig.workers.requests) == 3  # call 1 + call 2 with its one retry; nothing more


def test_live_trace_shows_the_provider_that_actually_answered(rig, tenant_a):
    rig.install(
        Workers(
            cf_tools(tool_call("f1", "get_shipment", shipment_number="SHP-1003")),
            RATE_LIMITED,
            RATE_LIMITED,
        ),
        ai_text("Shipment SHP-1003 is delayed."),
    )
    ev = rig.stream(tenant_a, "Where is SHP-1003?")
    models = [e for e in finished(ev) if e["kind"] == "model"]
    assert [(m["metadata"]["provider"], m["metadata"].get("fallback_used")) for m in models] == [
        ("cloudflare", None),
        ("gemini", True),
    ]
    assert ev[-1]["type"] == "run_completed"
    raw = json.dumps(ev)
    assert "capacity" not in raw and "3040" not in raw  # no raw provider error body
    final = ev[-1]["response"]["execution_trace"]
    assert model_rows(final) == [("cloudflare", False), ("gemini", True)]


# --- I: fallback around an approval trajectory ---------------------------------------------------
def test_fallback_around_approval_never_duplicates_the_action(rig, tenant_a, committed):
    rig.install(
        Workers(RATE_LIMITED, RATE_LIMITED),
        ai_tools(
            call(PROPOSE_CANCEL_ORDER, "g1", order_number="ORD-1004", reason="Customer request")
        ),
    )
    body = rig.post(tenant_a, "Cancel ORD-1004").json()
    assert body["action"]["status"] == "pending_approval"
    assert model_rows(body["execution_trace"]) == [("gemini", True)]
    action = body["action"]
    for _ in range(2):  # a repeated approve is idempotent
        rig.decide(tenant_a, action["id"], arguments_hash=action["arguments_hash"])
    with committed() as s:
        rows = s.scalars(select(ActionRequest.status)).all()
        executed = s.scalar(
            select(func.count())
            .select_from(AuditEvent)
            .where(AuditEvent.event_type == "action_succeeded")
        )
    assert rows == ["succeeded"] and executed == 1 and order_status(rig, tenant_a) == "cancelled"
    assert len(rig.gemini.invocations) == 1  # the proposal was generated once


# --- E/F/G: protocol, grounding and tenant/auth failures never fall back ---------------
def test_schema_invalid_tool_call_is_a_protocol_error_without_fallback(rig, tenant_a, tool_runs):
    bad = {"id": "b1", "type": "function", "function": {"name": "get_order", "arguments": "{oops"}}
    rig.install(Workers(cf_tools(bad)), ai_text("should never be used"))
    r = rig.post(tenant_a, "ORD-1001?")
    assert r.status_code == 502 and r.json()["error"]["code"] == "agent_protocol_error"
    assert rig.gemini.invocations == [] and tool_runs["n"] == 0


def test_textual_pseudo_tool_call_is_refused_without_fallback(rig, tenant_a, tool_runs):
    rig.install(
        Workers(
            cf_text(
                '{"type": "function", "name": "get_order", '
                '"parameters": {"order_number": "ORD-1001"}}'
            )
        ),
        ai_text("should never be used"),
    )
    r = rig.post(tenant_a, "ORD-1001?")
    assert r.status_code == 502 and r.json()["error"]["code"] == "agent_protocol_error"
    assert rig.gemini.invocations == [] and tool_runs["n"] == 0


def test_grounding_error_does_not_fall_back(rig, tenant_b):
    rig.install(
        Workers(
            cf_tools(tool_call("r1", "search_policy_knowledge", query="compensation")),
            cf_text("Credit applies [policy://invented/v9#chunk-1]."),
        ),
        ai_text("should never be used"),
    )
    r = rig.post(tenant_b, "What compensation applies?")
    assert r.status_code == 502 and r.json()["error"]["code"] == "agent_grounding_error"
    assert rig.gemini.invocations == []


def test_provider_auth_failure_does_not_fall_back(rig, tenant_a):
    rig.install(
        Workers(httpx.Response(401, json={"errors": [{"message": "bad token"}]})), ai_text("x")
    )
    r = rig.post(tenant_a, "hello")
    assert r.json()["error"]["code"] == "llm_auth_failed" and rig.gemini.invocations == []


def test_tenant_thread_conflict_never_reaches_any_provider(rig, tenant_a, tenant_b):
    rig.install(Workers(cf_text("A")), ai_text("never"))
    thread = "shared-thread"
    key = rig.assistant.thread_config(AgentContext(tenant_a.tenant_id), thread)
    rig.assistant.run("hi", AgentContext(tenant_a.tenant_id), thread_id=thread)
    # the same checkpoint key with ANOTHER tenant's runtime context is refused before any call
    graph = rig.assistant.graph
    out = graph.invoke({"messages": []}, key, context=AgentContext(tenant_b.tenant_id))
    assert out["error"]["code"] == "agent_thread_conflict"
    assert len(rig.workers.requests) == 1 and rig.gemini.invocations == []


# --- J: both providers unavailable ---------------------------------------------------------------
def test_both_providers_unavailable_is_the_existing_safe_error(rig, tenant_a):
    rig.install(Workers(RATE_LIMITED, RATE_LIMITED), RuntimeError("x"), RuntimeError("y"))
    rig.gemini.script[:] = [ConnectionError("down"), ConnectionError("down")]
    r = rig.post(tenant_a, "hello")
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_unavailable"
    assert "capacity" not in r.text


def test_without_fallback_a_rate_limit_is_reported_safely(rig, tenant_a):
    rig.install(Workers(RATE_LIMITED, RATE_LIMITED), fallback=False)
    r = rig.post(tenant_a, "hello")
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_rate_limited"
    with pytest.raises(AssistantError):
        rig.install(Workers(RATE_LIMITED, RATE_LIMITED), fallback=False).assistant.run(
            "hello", AgentContext(tenant_a.tenant_id), thread_id=str(uuid.uuid4())[:8]
        )
