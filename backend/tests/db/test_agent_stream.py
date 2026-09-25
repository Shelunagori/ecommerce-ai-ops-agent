"""Live execution trace: ``POST /api/agent/messages/stream`` and the decision streams.

Real PostgreSQL, real tools/actions, scripted model (same harness as test_agent_api). Every
assertion is about what the backend EMITS while the run executes: real boundaries only, in
order, safe fields only, converging on the final ``execution_trace``.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest
from sqlalchemy import func, select

from app.actions.capability import PROPOSE_CANCEL_ORDER
from app.agent.events import RunEmitter
from app.agent.llm.errors import LLMTimeoutError
from app.agent.prompts import graph_agent
from app.models import ActionRequest, AuditEvent, Order
from tests.assistant.fakes import ai_text, ai_tools, call
from tests.db.test_agent_api import V2, Harness, order_status, propose_cancel
from tests.rag.fakes import chunk, search

CHUNK_TEXT = chunk().content  # raw retrieved policy text: must never be streamed
SAFE_META_KEYS = {
    "call",
    "provider",
    "tool",
    "outcome",
    "duration_ms",
    "result_count",
    "retriever",
    "retrieval_mode",
    "as_of",
    "citations_verified",
    "action_type",
    "action_status",
    "decision",
    "failure_code",
    "error_code",
    "audit_recorded",
    "durable",
    "model_calls",
    "tool_calls",
    "citations",
}


def parse_sse(raw: str) -> list[dict]:
    events = []
    for block in raw.split("\n\n"):
        data = [line[6:] for line in block.split("\n") if line.startswith("data: ")]
        if data:
            events.append(json.loads("\n".join(data)))
    return events


class StreamHarness(Harness):
    def stream(self, tenant, text, thread="thread-1"):
        r = self.client.post(
            "/api/agent/messages/stream",
            json={"text": text, "thread_id": thread},
            headers={"X-Tenant-ID": str(tenant.tenant_id), "X-Request-ID": "req-stream-1"},
        )
        return r

    def events(self, tenant, text, thread="thread-1"):
        r = self.stream(tenant, text, thread)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/event-stream")
        return parse_sse(r.text)

    def decide_stream(self, tenant, action_id, decision="approve", **body):
        r = self.client.post(
            f"/api/agent/actions/{action_id}/{decision}/stream",
            json=body,
            headers={"X-Tenant-ID": str(tenant.tenant_id)},
        )
        assert r.status_code == 200, r.text
        return parse_sse(r.text)


@pytest.fixture
def s(committed):
    return StreamHarness(committed)


def steps(events, *types):
    return [e for e in events if e["type"] in types]


def finished(events):
    """Finished steps in emission order (what the final trace must equal)."""
    return [
        e
        for e in events
        if e["type"] in ("step_completed", "step_failed", "approval_required", "approval_resolved")
    ]


def as_trace(events):
    return [(e["kind"], e["label"], e["status"], e.get("detail"), e["metadata"]) for e in events]


def final_as_trace(trace):
    return [(t["kind"], t["label"], t["status"], t["detail"], t["metadata"]) for t in trace]


def terminal(events):
    return events[-1]


def assert_started_before_finished(events):
    started = {e["step_id"]: e["sequence"] for e in steps(events, "step_started")}
    for e in finished(events):
        if e["step_id"] in started:
            assert started[e["step_id"]] < e["sequence"]
    ended = {e["step_id"] for e in finished(events)}
    assert set(started) <= ended, "every started step reaches a final state"


# --- 1: request first ------------------------------------------------------------------------
def test_run_started_then_request_is_the_first_step(s, tenant_a):
    s.script(ai_tools(call("get_order", order_number="ORD-1001")), ai_text("Delivered."))
    ev = s.events(tenant_a, "Show me order ORD-1001")
    assert ev[0]["type"] == "run_started"
    assert [c["kind"] for c in ev[0]["capabilities"]] == [
        "commerce_tool",
        "retrieval",
        "grounding",
        "action_proposal",
        "response",
    ]
    assert (ev[1]["type"], ev[1]["kind"], ev[1]["status"]) == ("step_started", "request", "running")
    assert (ev[2]["type"], ev[2]["kind"], ev[2]["status"]) == (
        "step_completed",
        "request",
        "completed",
    )
    assert [e["sequence"] for e in ev] == list(range(1, len(ev) + 1))
    assert len({e["run_id"] for e in ev}) == 1


# --- 2/3/4: started before completed, measured durations -------------------------------------
def test_model_and_tool_steps_start_before_they_complete(s, tenant_a):
    s.script(ai_tools(call("get_order", order_number="ORD-1001")), ai_text("Delivered."))
    ev = s.events(tenant_a, "Show me order ORD-1001")
    assert_started_before_finished(ev)
    model_started = [e for e in steps(ev, "step_started") if e["kind"] == "model"]
    assert [e["detail"] for e in model_started] == ["Deciding the next step"] * 2
    assert [e["metadata"] for e in model_started] == [
        {"call": 1, "provider": "fake"},
        {"call": 2, "provider": "fake"},
    ]
    done = [e for e in steps(ev, "step_completed") if e["kind"] in ("model", "commerce_tool")]
    assert [(e["kind"], e["label"]) for e in done] == [
        ("model", "Agent orchestration"),
        ("commerce_tool", "Tool: get_order"),
        ("model", "Agent orchestration"),
    ]
    assert all(e["duration_ms"] is not None and e["duration_ms"] >= 0 for e in done)
    response = terminal(ev)["response"]
    assert done[1]["duration_ms"] == response["tool_calls"][0]["duration_ms"]  # measured once


def test_retrieval_starts_before_it_completes(s, tenant_b):
    s.script(ai_tools(search()), ai_text(f"15% store credit [{V2}]."))
    ev = s.events(tenant_b, "What compensation applies?")
    assert_started_before_finished(ev)
    started = next(e for e in steps(ev, "step_started") if e["kind"] == "retrieval")
    done = next(e for e in steps(ev, "step_completed") if e["kind"] == "retrieval")
    assert started["detail"] == "Searching tenant-scoped policy knowledge"
    assert done["metadata"]["result_count"] == 1 and done["detail"] == "1 eligible policy section"
    assert done["metadata"]["retriever"] == "semantic-pgvector-v1"


# --- 5/6/7: only what actually ran; unused capabilities skipped only at the end ---------------
def test_commerce_only_run_never_claims_rag_or_grounding(s, tenant_a):
    s.script(ai_tools(call("get_shipment", shipment_number="SHP-1003")), ai_text("Delayed."))
    ev = s.events(tenant_a, "Where is SHP-1003?")
    ran = {e["kind"] for e in steps(ev, "step_started", "step_completed", "step_failed")}
    assert "retrieval" not in ran and "grounding" not in ran
    skipped = steps(ev, "step_skipped")
    assert {(e["kind"], e["detail"]) for e in skipped} == {
        ("retrieval", "Not used in this run"),
        ("grounding", "Not used in this run"),
        ("action_proposal", "Not used in this run"),
    }
    # skipped only once the run is over: after the response step, right before run_completed
    response_seq = next(e["sequence"] for e in ev if e.get("kind") == "response")
    assert all(e["sequence"] > response_seq for e in skipped)
    assert terminal(ev)["type"] == "run_completed"


def test_rag_only_run_never_claims_a_commerce_tool(s, tenant_b):
    s.script(ai_tools(search()), ai_text(f"15% store credit [{V2}]."))
    ev = s.events(tenant_b, "What compensation applies?")
    ran = {e["kind"] for e in steps(ev, "step_started", "step_completed", "step_failed")}
    assert "commerce_tool" not in ran and {"retrieval", "grounding"} <= ran
    assert ("commerce_tool", "Not used in this run") in {
        (e["kind"], e["detail"]) for e in steps(ev, "step_skipped")
    }


def test_grounding_is_reported_only_when_it_ran(s, tenant_b):
    s.script(ai_text("Hello! How can I help?"))
    ev = s.events(tenant_b, "hi")
    assert not [e for e in finished(ev) if e["kind"] == "grounding"]
    s.script(ai_tools(search()), ai_text(f"15% store credit [{V2}]."))
    ev = s.events(tenant_b, "compensation?", thread="t2")
    g = [e for e in ev if e.get("kind") == "grounding" and e["type"] != "step_skipped"]
    assert [e["type"] for e in g] == ["step_started", "step_completed"]
    assert g[1]["metadata"] == {"citations_verified": 1}


# --- 8: mixed order ----------------------------------------------------------------------------
def test_mixed_run_streams_steps_in_the_executed_order(s, tenant_b):
    s.script(
        ai_tools(call("get_shipment", shipment_number="SHP-1003")),
        ai_tools(search()),
        ai_text(f"Delayed; 15% store credit [{V2}]."),
    )
    ev = s.events(tenant_b, "Where is SHP-1003, and what compensation applies?")
    assert [e["kind"] for e in finished(ev)] == [
        "request",
        "model",
        "commerce_tool",
        "model",
        "retrieval",
        "model",
        "grounding",
        "response",
    ]
    # one running step at a time: each start is followed by its own end before the next start
    open_steps = 0
    for e in ev:
        if e["type"] == "step_started":
            open_steps += 1
            assert open_steps == 1
        elif e["type"] in ("step_completed", "step_failed") and e["step_id"] in {
            x["step_id"] for x in steps(ev, "step_started")
        }:
            open_steps -= 1
    model_details = [e["detail"] for e in finished(ev) if e["kind"] == "model"]
    assert model_details == [
        "Requested commerce tool: get_shipment",
        "Requested policy retrieval",
        "Composed the final answer",
    ]


# --- 9/10/11: failures are red, the run ends with run_failed ----------------------------------
def test_unknown_tool_is_a_failed_step(s, tenant_a):
    s.script(ai_tools(call("no_such_tool")), ai_text("Sorry."))
    ev = s.events(tenant_a, "x")
    failed = steps(ev, "step_failed")
    assert [(e["kind"], e["status"], e["metadata"]["outcome"]) for e in failed] == [
        ("commerce_tool", "failed", "unknown_tool")
    ]
    assert terminal(ev)["type"] == "run_completed"  # the model explained; the run succeeded


def test_retrieval_failure_is_red_and_stops_the_run(s, tenant_b):
    s.retriever.script.append(RuntimeError("vector index offline at postgresql://secret"))
    s.script(ai_tools(search()), ai_text("unused"))
    raw = s.stream(tenant_b, "compensation?").text
    ev = parse_sse(raw)
    failed = steps(ev, "step_failed")
    assert [(e["kind"], e["detail"]) for e in failed] == [
        ("retrieval", "Policy retrieval service was unavailable")
    ]
    skipped = {(e["kind"], e["detail"]) for e in steps(ev, "step_skipped")}
    assert ("grounding", "Not run: the run stopped") in skipped
    assert ("response", "Not run: the run stopped") in skipped
    assert not [e for e in steps(ev, "step_completed") if e["kind"] in ("grounding", "response")]
    end = terminal(ev)
    assert end["type"] == "run_failed" and end["error"]["code"] == "agent_retrieval_error"
    assert end["error"]["status"] == 503 and "response" not in end
    assert "vector index offline" not in raw and "postgresql://" not in raw


def test_model_failure_is_a_failed_model_step_and_run_failed(s, tenant_a):
    s.script(LLMTimeoutError(), LLMTimeoutError())
    ev = s.events(tenant_a, "hello")
    failed = steps(ev, "step_failed")
    assert [(e["kind"], e["detail"]) for e in failed] == [
        ("model", "Model provider call failed (llm_timeout)")
    ]
    end = terminal(ev)
    assert (end["type"], end["error"]["code"], end["error"]["status"]) == (
        "run_failed",
        "llm_timeout",
        504,
    )


def test_invalid_input_fails_the_request_step(s, tenant_a):
    ev = s.events(tenant_a, "   ")
    assert [(e["kind"], e["status"]) for e in steps(ev, "step_failed")] == [("request", "failed")]
    assert terminal(ev)["error"]["code"] == "agent_input_invalid"


# --- 12/25: run_completed carries the normal response; the JSON endpoint is unchanged ----------
def test_run_completed_carries_the_same_response_as_the_json_endpoint(s, tenant_a):
    script = (ai_tools(call("get_order", order_number="ORD-1001")), ai_text("ORD-1001 delivered."))
    s.script(*script)
    streamed = terminal(s.events(tenant_a, "Show me order ORD-1001", thread="a"))["response"]
    s.script(*script)
    plain = s.post(tenant_a, "Show me order ORD-1001", thread="a2").json()

    def strip(body):
        body = json.loads(json.dumps(body))
        body.pop("thread_id")
        body.pop("duration_ms")
        for t in body["tool_calls"]:
            t.pop("duration_ms")
        for t in body["execution_trace"]:
            t["metadata"].pop("duration_ms", None)
        return body

    assert set(streamed) == set(plain)
    assert strip(streamed) == strip(plain)


# --- 13-17: safe fields only ------------------------------------------------------------------
def test_stream_contains_only_safe_fields(s, tenant_b, db_engine):
    s.script(
        ai_tools(call("get_shipment", shipment_number="SHP-1003")),
        ai_tools(search("SECRET-QUERY-TEXT")),
        ai_text(f"Delayed; 15% store credit [{V2}]."),
    )
    raw = s.stream(tenant_b, "Where is SHP-1003, and what compensation applies?").text
    ev = parse_sse(raw)
    step_events = [e for e in ev if e["type"] not in ("run_completed",)]
    step_raw = json.dumps(step_events)
    assert graph_agent.SYSTEM_PROMPT[:40] not in raw  # no prompt
    assert "SELECT" not in raw and "FROM " not in raw  # no SQL
    url = db_engine.url.render_as_string(hide_password=False)
    assert url not in raw and "postgresql" not in raw  # no database URL
    assert CHUNK_TEXT not in raw  # no raw retrieved policy text
    assert "embedding" not in step_raw and "[0." not in step_raw  # no vectors
    assert "SECRET-QUERY-TEXT" not in raw  # no model query / tool arguments
    assert "SHP-1003" not in step_raw  # tool arguments are not echoed in steps
    assert str(tenant_b.tenant_id) not in raw  # no tenant id
    for e in step_events:
        assert set(e.get("metadata", {})) <= SAFE_META_KEYS, e
        assert set(e) <= {
            "type",
            "run_id",
            "sequence",
            "elapsed_ms",
            "step_id",
            "kind",
            "label",
            "status",
            "detail",
            "metadata",
            "duration_ms",
            "capabilities",
            "next_steps",
            "error",
        }


# --- 18: tenant isolation ----------------------------------------------------------------------
def test_stream_is_tenant_scoped(s, tenant_a, tenant_b):
    s.script(ai_text("Answer for A."))
    s.events(tenant_a, "hello", thread="shared")
    s.script(ai_text("Answer for B."))
    s.events(tenant_b, "hello", thread="shared")
    history_a = s.get(tenant_a, "/api/agent/threads/shared/messages").json()["messages"]
    history_b = s.get(tenant_b, "/api/agent/threads/shared/messages").json()["messages"]
    assert [m["content"] for m in history_a] == ["hello", "Answer for A."]
    assert [m["content"] for m in history_b] == ["hello", "Answer for B."]
    # unknown tenant: refused BEFORE streaming, with the normal JSON error
    r = s.client.post(
        "/api/agent/messages/stream",
        json={"text": "hi", "thread_id": "x"},
        headers={"X-Tenant-ID": str(uuid.uuid4())},
    )
    assert r.status_code in (403, 404) and r.headers["content-type"].startswith("application/json")


def test_rate_limit_is_checked_before_streaming(s, tenant_a):
    s.app.state.settings.agent_rate_limit_per_minute = 1
    s.script(ai_text("one"), ai_text("two"))
    assert s.stream(tenant_a, "one", thread="r1").status_code == 200
    r = s.stream(tenant_a, "two", thread="r2")
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"


# --- 20/21: approval pause and live resume ------------------------------------------------------
def test_approval_required_only_for_a_real_pending_action(s, tenant_a):
    s.script(
        ai_tools(call(PROPOSE_CANCEL_ORDER, "p1", order_number="ORD-9999", reason="x")),
        ai_text("That order does not exist."),
    )
    ev = s.events(tenant_a, "Cancel ORD-9999", thread="refused")
    assert not steps(ev, "approval_required")
    proposal = next(e for e in finished(ev) if e["kind"] == "action_proposal")
    assert proposal["status"] == "rejected"  # refused by a rule: not an error, not a pause

    s.script(propose_cancel())
    ev = s.events(tenant_a, "Cancel ORD-1004")
    [approval] = steps(ev, "approval_required")
    assert (approval["kind"], approval["status"], approval["label"]) == (
        "approval",
        "waiting",
        "Human approval required",
    )
    assert [n["kind"] for n in approval["next_steps"]] == ["action_execution", "response"]
    after = [e for e in ev if e["sequence"] > approval["sequence"]]
    assert [e.get("kind") for e in after if e["type"] != "step_skipped"][:2] == [
        "checkpoint",
        "response",
    ]
    assert terminal(ev)["type"] == "run_completed"
    assert terminal(ev)["response"]["action"]["status"] == "pending_approval"
    assert order_status(s, tenant_a) == "processing"  # nothing changed while waiting


def test_approve_stream_reports_decision_execution_and_response_in_order(s, tenant_a):
    s.script(propose_cancel())
    action = terminal(s.events(tenant_a, "Cancel ORD-1004"))["response"]["action"]
    ev = s.decide_stream(tenant_a, action["id"], arguments_hash=action["arguments_hash"])
    assert ev[0]["type"] == "run_started"
    assert [c["kind"] for c in ev[0]["capabilities"]] == [
        "approval",
        "action_execution",
        "response",
    ]
    body = [e for e in ev[1:-1]]
    assert [(e["type"], e.get("kind")) for e in body] == [
        ("approval_resolved", "approval"),
        ("step_started", "action_execution"),
        ("step_completed", "action_execution"),
        ("step_completed", "response"),
    ]
    assert body[0]["label"] == "Approved by a human" and body[0]["status"] == "completed"
    assert body[2]["metadata"]["audit_recorded"] is True
    assert body[3]["label"] == "Outcome recorded"
    end = terminal(ev)
    assert end["type"] == "run_completed" and end["response"]["action"]["status"] == "succeeded"
    assert order_status(s, tenant_a) == "cancelled"


def test_reject_stream_skips_execution(s, tenant_a):
    s.script(propose_cancel())
    action = terminal(s.events(tenant_a, "Cancel ORD-1004"))["response"]["action"]
    ev = s.decide_stream(tenant_a, action["id"], "reject", arguments_hash=action["arguments_hash"])
    resolved = steps(ev, "approval_resolved")[0]
    assert (resolved["label"], resolved["status"]) == ("Rejected by a human", "rejected")
    assert not [e for e in steps(ev, "step_started") if e["kind"] == "action_execution"]
    assert ("action_execution", "Not executed: nothing was changed") in {
        (e["kind"], e["detail"]) for e in steps(ev, "step_skipped")
    }
    assert order_status(s, tenant_a) == "processing"


# --- 22: failed execution is red -----------------------------------------------------------------
def test_failed_execution_is_a_failed_step(s, tenant_a, committed):
    s.script(propose_cancel())
    action = terminal(s.events(tenant_a, "Cancel ORD-1004"))["response"]["action"]
    with committed() as session:
        session.execute(
            Order.__table__.update()
            .where(Order.tenant_id == tenant_a.tenant_id, Order.order_number == "ORD-1004")
            .values(status="shipped")
        )
    ev = s.decide_stream(tenant_a, action["id"], arguments_hash=action["arguments_hash"])
    [failed] = steps(ev, "step_failed")
    assert (failed["kind"], failed["status"]) == ("action_execution", "failed")
    assert failed["metadata"]["failure_code"] == "order_not_cancellable"
    assert "audit_recorded" not in failed["metadata"]
    assert terminal(ev)["response"]["action"]["status"] == "failed"


# --- 23: the stream converges on the final trace --------------------------------------------------
@pytest.mark.parametrize(
    "script",
    [
        "commerce",
        "rag",
        "mixed",
        "failed_tool",
        "approval",
    ],
)
def test_finished_stream_steps_equal_the_final_trace(s, tenant_b, script):
    scripts = {
        "commerce": (
            ai_tools(call("get_shipment", shipment_number="SHP-1003")),
            ai_text("Delayed."),
        ),
        "rag": (ai_tools(search()), ai_text(f"15% store credit [{V2}].")),
        "mixed": (
            ai_tools(call("get_shipment", shipment_number="SHP-1003")),
            ai_tools(search()),
            ai_text(f"Delayed; 15% store credit [{V2}]."),
        ),
        "failed_tool": (ai_tools(call("no_such_tool")), ai_text("Sorry.")),
        "approval": (propose_cancel("ORD-1004"),),
    }
    s.script(*scripts[script])
    ev = s.events(tenant_b, "run", thread=f"c-{script}")
    trace = terminal(ev)["response"]["execution_trace"]
    assert as_trace(finished(ev)) == final_as_trace(trace)


def test_resumed_stream_plus_paused_trace_equals_the_final_trace(s, tenant_a):
    s.script(propose_cancel())
    paused = terminal(s.events(tenant_a, "Cancel ORD-1004"))["response"]
    before = [t for t in paused["execution_trace"] if t["status"] != "waiting"]
    before = [t for t in before if t["kind"] != "checkpoint"]
    ev = s.decide_stream(
        tenant_a, paused["action"]["id"], arguments_hash=paused["action"]["arguments_hash"]
    )
    final = terminal(ev)["response"]["execution_trace"]
    live = as_trace(finished(ev))

    def no_run_total(rows):  # the response step reports the TOTAL duration of each run
        return [
            (
                k,
                lbl,
                st,
                d,
                {x: v for x, v in m.items() if x != "duration_ms"} if k == "response" else m,
            )
            for k, lbl, st, d, m in rows
        ]

    assert no_run_total(final_as_trace(before) + live) == no_run_total(final_as_trace(final))


# --- 24: a broken event sink never changes the run ---------------------------------------------
def test_broken_sink_cannot_alter_business_execution(s, tenant_a, committed, monkeypatch):
    calls = {"n": 0}

    def exploding(_event):
        calls["n"] += 1
        raise RuntimeError("client went away")

    s.script(propose_cancel())
    runtime = s.app.state.agent_runtime
    from app.agent.context import AgentContext

    ctx = AgentContext(tenant_a.tenant_id, "req-sink")
    paused = runtime.assistant.run(
        "Cancel ORD-1004", ctx, thread_id="sink", events=RunEmitter(exploding)
    )
    assert paused.action.status == "pending_approval" and calls["n"] > 0
    done = runtime.assistant.resume(
        ctx,
        thread_id="sink",
        action_id=uuid.UUID(paused.action.id),
        decision="approve",
        decided_by="approver",
        expected_hash=paused.action.arguments_hash,
        events=RunEmitter(exploding),
    )
    assert done.action.status == "succeeded" and order_status(s, tenant_a) == "cancelled"
    with committed() as session:
        executed = session.scalar(
            select(func.count())
            .select_from(AuditEvent)
            .where(AuditEvent.event_type == "action_succeeded")
        )
    assert executed == 1
    assert [t.kind for t in done.execution_trace][-1] == "response"


# --- disconnect: the run is not tied to the connection; nothing runs twice -----------------------
def test_disconnect_during_approve_stream_does_not_duplicate_the_write(s, tenant_a, committed):
    s.script(propose_cancel())
    action = terminal(s.events(tenant_a, "Cancel ORD-1004"))["response"]["action"]
    with s.client.stream(
        "POST",
        f"/api/agent/actions/{action['id']}/approve/stream",
        json={"arguments_hash": action["arguments_hash"]},
        headers={"X-Tenant-ID": str(tenant_a.tenant_id)},
    ) as r:
        assert r.status_code == 200
        next(r.iter_lines())  # read one line, then the client goes away
    deadline = time.monotonic() + 10
    while order_status(s, tenant_a) != "cancelled" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert order_status(s, tenant_a) == "cancelled"  # the run finished server-side
    # a manual retry (JSON endpoint) is idempotent: no second execution
    again = s.decide(tenant_a, action["id"], arguments_hash=action["arguments_hash"])
    assert again.status_code == 200 and again.json()["action"]["status"] == "succeeded"
    with committed() as session:
        rows = session.scalars(select(ActionRequest.status)).all()
        executed = session.scalar(
            select(func.count())
            .select_from(AuditEvent)
            .where(AuditEvent.event_type == "action_succeeded")
        )
    assert rows == ["succeeded"] and executed == 1


# --- 19: public demo stays read-only in the live trace ----------------------------------------
def test_public_demo_stream_has_no_action_capability(committed, tenant_b):
    from tests.db.test_public_demo import ANON, h, token
    from tests.db.test_public_demo import Harness as DemoHarness

    demo = DemoHarness(committed)
    demo.script(
        ai_tools(call(PROPOSE_CANCEL_ORDER, "p1", order_number="ORD-1004", reason="x")),
        ai_text("The public demo is read-only; I cannot cancel orders."),
    )
    before = demo.client.get("/api/me", headers=h(token(ANON))).json()
    assert before["public_demo"] is True
    r = demo.client.post(
        "/api/agent/messages/stream",
        json={"text": "Cancel ORD-1004", "thread_id": "demo"},
        headers=h(token(ANON), tenant_b),
    )
    assert r.status_code == 200, r.text
    ev = parse_sse(r.text)
    assert "action_proposal" not in [c["kind"] for c in ev[0]["capabilities"]]
    request = next(e for e in finished(ev) if e["kind"] == "request")
    assert request["detail"].endswith("read-only public demo (no action tools)")
    kinds = {e.get("kind") for e in ev}
    assert "action_proposal" not in kinds and "approval" not in kinds
    assert not steps(ev, "approval_required")
    tool = next(e for e in finished(ev) if e["kind"] == "commerce_tool")
    assert (tool["status"], tool["metadata"]["outcome"]) == ("failed", "unknown_tool")
    assert terminal(ev)["response"]["action"] is None
    with committed() as session:
        assert session.scalar(select(func.count()).select_from(ActionRequest)) == 0


# --- real time: a step is reported running BEFORE its work executes ---------------------------
def test_steps_are_reported_running_while_the_work_executes(s, tenant_b):
    from app.agent.context import AgentContext

    events = []
    seen_during = {}

    def model_turn(messages):
        seen_during.setdefault("model", (events[-1].type, events[-1].kind))
        if len(seen_during) == 1:
            return ai_tools(search())
        return ai_text(f"15% store credit [{V2}].")

    class WatchingRetriever:
        def retrieve(self, query, context, *, as_of=None, limit=5):
            seen_during["retrieval"] = (events[-1].type, events[-1].kind)
            seen_during.setdefault("model_calls", 0)
            return s.retriever.retrieve(query, context, as_of=as_of, limit=limit)

    s.script(model_turn, model_turn, retriever=WatchingRetriever())
    assistant = s.app.state.agent_runtime.assistant
    result = assistant.run(
        "compensation?",
        AgentContext(tenant_b.tenant_id, "req-rt"),
        thread_id="rt",
        events=RunEmitter(events.append),
    )
    assert result.answer.startswith("15%")
    assert seen_during["model"] == ("step_started", "model")
    assert seen_during["retrieval"] == ("step_started", "retrieval")


def test_public_demo_cannot_use_the_decision_streams(committed, tenant_b):
    from tests.db.test_public_demo import ANON, h, token
    from tests.db.test_public_demo import Harness as DemoHarness

    demo = DemoHarness(committed)
    r = demo.client.post(
        f"/api/agent/actions/{uuid.uuid4()}/approve/stream",
        json={},
        headers=h(token(ANON), tenant_b),
    )
    assert (r.status_code, r.json()["error"]["code"]) == (403, "public_demo_read_only")
