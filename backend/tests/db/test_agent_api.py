"""Agent API (Phase 6) on real PostgreSQL with committed writes; scripted model, fake
retriever, in-memory checkpoints (the durable saver is covered in test_durable_checkpoint.py).
Demo auth mode here; the Supabase JWT boundary is in test_auth.py."""

import json
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import func, select

import app.db.session as db_session_module
from app.actions.capability import PROPOSE_CANCEL_ORDER, PROPOSE_STORE_CREDIT
from app.actions.service import ActionService
from app.agent.assistant import AssistantLimits
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.profile import AGENT_PROFILE
from app.agent.prompts import graph_agent
from app.agent.tools import ToolDependencies, build_commerce_tools
from app.api.agent_runtime import AgentRuntime
from app.main import create_app
from app.models import Order, StoreCreditTransaction
from app.observability.runs import RunRecorder
from tests.assistant.fakes import ai_text, ai_tools, call, make_provider
from tests.conftest import make_settings
from tests.db.conftest import fixed_clock
from tests.rag.fakes import FakeRetriever, search

T0 = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
V2 = "policy://delayed-shipment-compensation/v2#chunk-2"
REQ = httpx.Request("POST", "http://localhost:11434/api/chat")


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now


class Harness:
    def __init__(self, committed):
        self.committed = committed
        self.clock = Clock()
        self.actions = ActionService(committed, clock=self.clock)
        self.saver = InMemorySaver()
        self.tools = build_commerce_tools(
            ToolDependencies(session_scope=db_session_module.read_only_session, clock=fixed_clock)
        )
        self.app = create_app(make_settings())
        self.client = TestClient(self.app)
        self.retriever = FakeRetriever()
        self.script()

    def script(self, *steps, retriever=None):
        provider, self.model = make_provider(*steps)
        assistant = CommerceGraphAssistant(
            provider,
            tools=self.tools,
            limits=AssistantLimits(),
            checkpointer=self.saver,
            retriever=retriever or self.retriever,
            profile=AGENT_PROFILE,
            actions=self.actions,
            run_recorder=RunRecorder(self.committed),
        )
        self.app.state.agent_runtime = AgentRuntime(assistant, self.actions)

    def post(self, tenant, text, thread="thread-1", **extra):
        return self.client.post(
            "/api/agent/messages",
            json={"text": text, "thread_id": thread, **extra},
            headers={"X-Tenant-ID": str(tenant.tenant_id), "X-Request-ID": "req-api-1"},
        )

    def get(self, tenant, path):
        return self.client.get(path, headers={"X-Tenant-ID": str(tenant.tenant_id)})

    def decide(self, tenant, action_id, decision="approve", **body):
        return self.client.post(
            f"/api/agent/actions/{action_id}/{decision}",
            json=body,
            headers={"X-Tenant-ID": str(tenant.tenant_id)},
        )


@pytest.fixture
def h(committed):
    return Harness(committed)


def order_status(h, tenant, number="ORD-1004"):
    with h.committed() as s:
        return s.scalar(
            select(Order.status).where(
                Order.tenant_id == tenant.tenant_id, Order.order_number == number
            )
        )


def propose_cancel(number="ORD-1004"):
    return ai_tools(
        call(PROPOSE_CANCEL_ORDER, "p1", order_number=number, reason="Customer request")
    )


# --- chat --------------------------------------------------------------------------------------
def test_chat_request_returns_the_application_contract(h, tenant_a):
    h.script(
        ai_tools(call("get_order", order_number="ORD-1001")), ai_text("ORD-1001 was delivered.")
    )
    r = h.post(tenant_a, "Show me order ORD-1001")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {
        "thread_id",
        "answer",
        "prompt_version",
        "model_calls",
        "duration_ms",
        "tool_calls",
        "retrievals",
        "citations",
        "action",
    }
    assert (
        body["answer"] == "ORD-1001 was delivered." and body["tool_calls"][0]["tool"] == "get_order"
    )
    assert body["prompt_version"] == "commerce-assistant-v3" and body["action"] is None
    raw = r.text
    assert str(tenant_a.tenant_id) not in raw and graph_agent.SYSTEM_PROMPT[:40] not in raw
    assert '"data"' not in raw  # no raw tool payloads


def test_thread_continuation_and_sanitised_history(h, tenant_a):
    h.script(ai_text("Hello!"), ai_text("Still here."))
    h.post(tenant_a, "hi")
    h.post(tenant_a, "are you there?")
    r = h.get(tenant_a, "/api/agent/threads/thread-1/messages")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", "hi"),
        ("assistant", "Hello!"),
        ("user", "are you there?"),
        ("assistant", "Still here."),
    ]
    assert r.json()["pending_action"] is None and "system" not in r.text


def test_policy_citations_render_contract(h, tenant_b):
    h.script(ai_tools(search()), ai_text(f"15% store credit [{V2}]."))
    body = h.post(tenant_b, "What compensation applies?").json()
    [c] = body["citations"]
    assert c == {
        "citation": V2,
        "title": "Delayed Shipment Compensation Policy",
        "document_key": "delayed-shipment-compensation",
        "version": 2,
        "section": "Delayed Shipment Compensation Policy > Compensation",
        "effective_from": "2026-08-15",
        "effective_to": None,
    }
    assert body["retrievals"][0]["citations"] == [V2] and "content" not in json.dumps(
        body["retrievals"]
    )


# --- approvals ----------------------------------------------------------------------------------
def test_pending_approval_then_approve(h, tenant_a):
    h.script(propose_cancel())
    body = h.post(tenant_a, "Cancel ORD-1004").json()
    action = body["action"]
    assert action["status"] == "pending_approval" and order_status(h, tenant_a) == "processing"
    history = h.get(tenant_a, "/api/agent/threads/thread-1/messages").json()
    assert history["pending_action"]["id"] == action["id"]
    listed = h.get(tenant_a, "/api/agent/actions?status=pending_approval").json()
    assert [a["id"] for a in listed] == [action["id"]]
    r = h.decide(tenant_a, action["id"], arguments_hash=action["arguments_hash"])
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["action"]["status"] == "succeeded" and out["answer"].startswith(
        "Done: order ORD-1004"
    )
    assert [e["event_type"] for e in out["action"]["events"]][:2] == [
        "action_requested",
        "approval_decided",
    ]
    assert order_status(h, tenant_a) == "cancelled"
    assert h.get(tenant_a, f"/api/agent/actions/{action['id']}").json()["status"] == "succeeded"


def test_reject(h, tenant_a):
    h.script(propose_cancel())
    action = h.post(tenant_a, "Cancel ORD-1004").json()["action"]
    out = h.decide(tenant_a, action["id"], "reject").json()
    assert out["action"]["status"] == "rejected" and "nothing was changed" in out["answer"]
    assert order_status(h, tenant_a) == "processing"


def test_expired_approval(h, tenant_a):
    h.script(propose_cancel())
    action = h.post(tenant_a, "Cancel ORD-1004").json()["action"]
    h.clock.now = T0 + timedelta(days=1)
    out = h.decide(tenant_a, action["id"], arguments_hash=action["arguments_hash"]).json()
    assert out["action"]["status"] == "expired" and order_status(h, tenant_a) == "processing"


def test_duplicate_approve_is_idempotent(h, tenant_a):
    h.script(
        ai_tools(search()),
        ai_tools(
            call(
                PROPOSE_STORE_CREDIT,
                "c1",
                customer_code="CUS-1002",
                amount="15.00",
                currency="USD",
                reason="Delay",
                order_number="ORD-1003",
                policy_citations=[V2],
            )
        ),
    )
    action = h.post(tenant_a, "Credit CUS-1002").json()["action"]
    first = h.decide(tenant_a, action["id"], arguments_hash=action["arguments_hash"])
    second = h.decide(tenant_a, action["id"], arguments_hash=action["arguments_hash"])
    assert first.status_code == second.status_code == 200
    assert first.json()["action"]["result"] == second.json()["action"]["result"]
    with h.committed() as s:
        assert s.scalar(select(func.count()).select_from(StoreCreditTransaction)) == 1


def test_approval_bound_to_the_shown_hash(h, tenant_a):
    h.script(propose_cancel())
    action = h.post(tenant_a, "Cancel ORD-1004").json()["action"]
    r = h.decide(tenant_a, action["id"], arguments_hash="0" * 64)
    assert r.status_code == 409 and r.json()["error"]["code"] == "arguments_hash_mismatch"
    assert order_status(h, tenant_a) == "processing"


def test_rejected_then_approve_is_refused(h, tenant_a):
    h.script(propose_cancel())
    action = h.post(tenant_a, "Cancel ORD-1004").json()["action"]
    h.decide(tenant_a, action["id"], "reject")
    r = h.decide(tenant_a, action["id"])
    assert r.status_code == 409 and r.json()["error"]["code"] == "approval_already_resolved"


def test_new_message_while_pending_is_409(h, tenant_a):
    h.script(propose_cancel(), ai_text("never"))
    h.post(tenant_a, "Cancel ORD-1004")
    r = h.post(tenant_a, "something else")
    assert r.status_code == 409 and r.json()["error"]["code"] == "agent_approval_pending"


# --- tenant isolation / validation / failures ---------------------------------------------------
def test_other_tenant_cannot_see_or_decide(h, tenant_a, tenant_b):
    h.script(propose_cancel())
    action = h.post(tenant_a, "Cancel ORD-1004").json()["action"]
    for r in (
        h.get(tenant_b, f"/api/agent/actions/{action['id']}"),
        h.decide(tenant_b, action["id"]),
        h.decide(tenant_b, action["id"], "reject"),
    ):
        assert r.status_code == 404 and r.json()["error"]["code"] == "action_not_found"
    assert h.get(tenant_b, "/api/agent/actions").json() == []
    hist = h.get(tenant_b, "/api/agent/threads/thread-1/messages").json()
    assert hist["messages"] == [] and hist["pending_action"] is None
    assert order_status(h, tenant_a) == "processing"


@pytest.mark.parametrize(
    "body",
    [
        {"text": "hi", "thread_id": "t", "tenant_id": "11111111-1111-4111-8111-111111111111"},
        {"text": "hi", "thread_id": "bad id!"},
        {"text": "", "thread_id": "t"},
        {"text": "x" * 4001, "thread_id": "t"},
    ],
)
def test_request_validation(h, tenant_a, body):
    r = h.client.post(
        "/api/agent/messages", json=body, headers={"X-Tenant-ID": str(tenant_a.tenant_id)}
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert "11111111" not in r.text


def test_missing_tenant_in_demo_mode(h):
    r = h.client.post("/api/agent/messages", json={"text": "hi", "thread_id": "t"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "tenant_context_missing"


def test_llm_failure_is_a_stable_envelope(h, tenant_a):
    h.script(httpx.ReadTimeout("slow", request=REQ), httpx.ReadTimeout("slow", request=REQ))
    r = h.post(tenant_a, "hi")
    assert r.status_code == 504 and r.json()["error"]["code"] == "llm_timeout"
    assert r.json()["request_id"] == "req-api-1"


def test_retrieval_failure_is_503_not_no_policy(h, tenant_b):
    h.script(
        ai_tools(search()), ai_text("From memory"), retriever=FakeRetriever([RuntimeError("down")])
    )
    r = h.post(tenant_b, "Policy?")
    assert r.status_code == 503 and r.json()["error"]["code"] == "agent_retrieval_error"
    assert "down" not in r.text


def test_unknown_action_id_is_404(h, tenant_a):
    r = h.get(tenant_a, f"/api/agent/actions/{uuid.uuid4()}")
    assert r.status_code == 404


def test_me_lists_demo_tenants(h):
    r = h.client.get("/api/me")
    assert r.status_code == 200
    body = r.json()
    assert body["auth_mode"] == "demo" and {m["slug"] for m in body["memberships"]} == {
        "northstar-commerce",
        "bluepeak-retail",
    }


# --- Phase 12: abuse protection -----------------------------------------------------------------
def test_agent_messages_are_rate_limited_per_user_and_tenant(h, tenant_a, tenant_b):
    from app.api.ratelimit import RateLimiter

    h.app.state.rate_limiter = RateLimiter(2)
    h.script(ai_text("one"), ai_text("two"), ai_text("other tenant"))
    assert h.post(tenant_a, "hi").status_code == 200
    assert h.post(tenant_a, "hi again").status_code == 200
    r = h.post(tenant_a, "third")
    assert (r.status_code, r.json()["error"]["code"]) == (429, "rate_limited")
    assert int(r.headers["retry-after"]) >= 1
    assert len(h.model.invocations) == 2  # refused before any model call
    assert h.post(tenant_b, "hello").status_code == 200  # separate bucket
