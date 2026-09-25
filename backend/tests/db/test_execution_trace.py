"""Agent Execution Trace: generated from the graph's execution records, ordered as executed,
and limited to safe operational metadata (no prompts, reasoning, SQL, secrets or tenant ids).
"""

import json

import pytest
from langchain_core.messages import AIMessage

from app.actions.capability import PROPOSE_CANCEL_ORDER, PROPOSE_STORE_CREDIT
from app.agent.prompts import graph_agent
from app.agent.trace import TraceEventKind
from tests.assistant.fakes import ai_text, ai_tools, call
from tests.db.test_agent_api import V2, Harness, propose_cancel
from tests.rag.fakes import search

SECRET_REASONING = "INTERNAL-REASONING-DO-NOT-SHOW"


@pytest.fixture
def h(committed):
    return Harness(committed)


def kinds(trace):
    return [e["kind"] for e in trace]


def assert_well_formed(trace):
    assert [e["sequence"] for e in trace] == list(range(1, len(trace) + 1))
    assert trace[0]["kind"] == "request" and trace[-1]["kind"] == "response"


def test_commerce_only_trace(h, tenant_a):
    h.script(ai_tools(call("get_order", order_number="ORD-1001")), ai_text("Delivered."))
    body = h.post(tenant_a, "Show me order ORD-1001").json()
    trace = body["execution_trace"]
    assert_well_formed(trace)
    assert kinds(trace) == ["request", "model", "commerce_tool", "model", "response"]
    tool = trace[2]
    assert tool["label"] == "Tool: get_order" and tool["status"] == "completed"
    assert tool["metadata"]["outcome"] == "success"
    assert tool["metadata"]["duration_ms"] == body["tool_calls"][0]["duration_ms"]  # measured
    assert trace[1]["metadata"] == {"call": 1, "provider": "fake"}
    assert trace[3]["detail"] == "Composed the final answer"
    assert trace[-1]["metadata"]["model_calls"] == body["model_calls"] == 2


def test_rag_only_trace(h, tenant_b):
    h.script(ai_tools(search()), ai_text(f"15% store credit [{V2}]."))
    body = h.post(tenant_b, "What compensation applies?").json()
    trace = body["execution_trace"]
    assert kinds(trace) == ["request", "model", "retrieval", "model", "grounding", "response"]
    retrieval = trace[2]["metadata"]
    assert retrieval["result_count"] == body["retrievals"][0]["result_count"]
    assert retrieval["retriever"] == "semantic-pgvector-v1"
    assert retrieval["retrieval_mode"] == "semantic (pgvector)"
    assert trace[4]["metadata"] == {"citations_verified": 1}


def test_mixed_trace_follows_the_executed_order(h, tenant_b):
    h.script(
        ai_tools(call("get_shipment", shipment_number="SHP-1003")),
        ai_tools(search()),
        ai_text(f"Delayed; 15% store credit [{V2}]."),
    )
    trace = h.post(tenant_b, "Where is SHP-1003 and what applies?").json()["execution_trace"]
    assert kinds(trace) == [
        "request",
        "model",
        "commerce_tool",
        "model",
        "retrieval",
        "model",
        "grounding",
        "response",
    ]
    assert [e["metadata"].get("call") for e in trace if e["kind"] == "model"] == [1, 2, 3]
    assert trace[2]["label"] == "Tool: get_shipment"


def test_retrieval_before_tool_is_reported_in_that_order(h, tenant_b):
    """Order comes from the recorded rounds, not from a fixed template."""
    h.script(ai_tools(search()), ai_text(f"Policy [{V2}]."))
    first = kinds(h.post(tenant_b, "policy?", thread="a").json()["execution_trace"])
    h.script(ai_tools(call("get_order", order_number="ORD-1001")), ai_text("ok"))
    second = kinds(h.post(tenant_b, "order?", thread="b").json()["execution_trace"])
    assert first.index("retrieval") < first.index("response") and "commerce_tool" not in first
    assert second.index("commerce_tool") < second.index("response") and "retrieval" not in second


def test_failed_tool_is_marked_failed(h, tenant_a):
    h.script(ai_tools(call("get_order", order_number="ORD-9999")), ai_text("Not found."))
    trace = h.post(tenant_a, "ORD-9999?").json()["execution_trace"]
    tool = next(e for e in trace if e["kind"] == "commerce_tool")
    assert tool["metadata"]["outcome"] == "not_found" and tool["status"] == "completed"
    h.script(ai_tools(call("no_such_tool")), ai_text("Sorry."))
    trace = h.post(tenant_a, "x", thread="t2").json()["execution_trace"]
    tool = next(e for e in trace if e["kind"] == "commerce_tool")
    assert (tool["status"], tool["metadata"]["outcome"]) == ("failed", "unknown_tool")


def test_approval_trace_then_decision(h, tenant_a):
    h.script(propose_cancel())
    body = h.post(tenant_a, "Cancel ORD-1004").json()
    trace = body["execution_trace"]
    assert kinds(trace) == [
        "request",
        "model",
        "action_proposal",
        "approval",
        "checkpoint",
        "response",
    ]
    approval = trace[3]
    assert (approval["label"], approval["status"]) == ("Human approval required", "waiting")
    assert trace[4]["metadata"] == {"durable": False}  # InMemorySaver in this test
    assert trace[-1]["status"] == "waiting"
    action = body["action"]
    decided = h.decide(tenant_a, action["id"], arguments_hash=action["arguments_hash"]).json()
    after = decided["execution_trace"]
    assert kinds(after) == [
        "request",
        "model",
        "action_proposal",
        "approval",
        "action_execution",
        "response",
    ]
    assert after[3]["label"] == "Approved by a human"
    assert (after[4]["status"], after[4]["metadata"]["action_status"]) == ("completed", "succeeded")
    # A repeated approve has no graph to resume: no invented trace.
    again = h.decide(tenant_a, action["id"], arguments_hash=action["arguments_hash"]).json()
    assert again["execution_trace"] is None


def test_rejection_has_no_execution_event(h, tenant_a):
    h.script(propose_cancel())
    action = h.post(tenant_a, "Cancel ORD-1004").json()["action"]
    after = h.decide(tenant_a, action["id"], "reject").json()["execution_trace"]
    assert "action_execution" not in kinds(after)
    approval = next(e for e in after if e["kind"] == "approval")
    assert (approval["status"], approval["detail"]) == ("rejected", "Nothing was changed")


def test_refused_proposal_is_shown_as_rejected(h, tenant_a):
    h.script(
        ai_tools(call(PROPOSE_CANCEL_ORDER, "p9", order_number="ORD-1003", reason="x")),
        ai_text("ORD-1003 has shipped and cannot be cancelled."),
    )
    trace = h.post(tenant_a, "Cancel ORD-1003").json()["execution_trace"]
    proposal = next(e for e in trace if e["kind"] == "action_proposal")
    assert proposal["status"] == "rejected"
    assert proposal["metadata"]["error_code"] == "order_not_cancellable"
    assert "approval" not in kinds(trace)


def test_trace_contains_only_safe_metadata(h, tenant_a):
    h.script(
        ai_tools(call("get_customer", customer_code="CUS-1002")),
        ai_tools(search()),
        AIMessage(
            content=f"<thinking>{SECRET_REASONING}</thinking>",
            tool_calls=[
                call(
                    PROPOSE_STORE_CREDIT,
                    "c1",
                    customer_code="CUS-1002",
                    amount="15.00",
                    currency="USD",
                    reason=f"Delay {SECRET_REASONING}",
                    order_number="ORD-1003",
                    policy_citations=[V2],
                )
            ],
        ),
    )
    body = h.post(tenant_a, "Credit CUS-1002 for the delay, my email is a@b.example").json()
    raw = json.dumps(body["execution_trace"])
    forbidden = [
        SECRET_REASONING,  # model reasoning / free text
        graph_agent.SYSTEM_PROMPT[:40],  # prompts
        "a@b.example",  # user text
        "CUS-1002",  # tool arguments
        "15.00",
        str(tenant_a.tenant_id),  # tenant ids
        "SELECT",  # SQL
        "postgresql",  # connection strings
        "Customers receive store credit",  # retrieved policy text
        "Bearer",
    ]
    for needle in forbidden:
        assert needle not in raw, needle
    allowed_keys = {
        "call",
        "provider",
        "tool",
        "outcome",
        "duration_ms",
        "result_count",
        "retriever",
        "retrieval_mode",
        "as_of",
        "action_type",
        "error_code",
        "action_status",
        "durable",
        "citations_verified",
        "model_calls",
        "tool_calls",
        "citations",
        "failure_code",
    }
    for event in body["execution_trace"]:
        assert set(event) == {"sequence", "kind", "label", "status", "detail", "metadata"}
        assert set(event["metadata"]) <= allowed_keys
        assert event["kind"] in {k.value for k in TraceEventKind}


def test_history_does_not_carry_traces(h, tenant_a):
    h.script(ai_text("Hello!"))
    h.post(tenant_a, "hi")
    history = h.get(tenant_a, "/api/agent/threads/thread-1/messages").json()
    assert "execution_trace" not in json.dumps(history)
