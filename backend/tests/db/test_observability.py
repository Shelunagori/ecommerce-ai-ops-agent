"""Durable observability and audit (Phase 5): transactional action audit events, best-effort
run records, correlation ids, and nothing sensitive stored."""

import json
import logging
import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

import app.db.session as db_session_module
from app.actions import errors as E  # noqa: N812
from app.actions.capability import PROPOSE_CANCEL_ORDER
from app.actions.service import ActionService
from app.agent.assistant import AssistantLimits
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.profile import AGENT_PROFILE
from app.agent.tools import ToolDependencies, build_commerce_tools
from app.core.config import Settings
from app.models import AgentRun, AuditEvent
from app.observability.runs import RunRecorder
from tests.assistant.fakes import ai_text, ai_tools, call, make_provider
from tests.db.conftest import fixed_clock
from tests.rag.fakes import FakeRetriever

T0 = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
SAFE_DETAIL_KEYS = {"status", "action_type", "arguments_hash", "decision", "failure_code"}


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def svc(committed, clock):
    return ActionService(committed, clock=clock)


@pytest.fixture
def ns(tenant_a):
    return AgentContext(tenant_a.tenant_id, "req-obs-1")


def agent(committed, svc, *script, recorder=True):
    provider, _ = make_provider(*script)
    tools = build_commerce_tools(
        ToolDependencies(session_scope=db_session_module.read_only_session, clock=fixed_clock)
    )
    return CommerceGraphAssistant(
        provider,
        tools=tools,
        limits=AssistantLimits(),
        checkpointer=InMemorySaver(),
        retriever=FakeRetriever(),
        profile=AGENT_PROFILE,
        actions=svc,
        run_recorder=RunRecorder(committed) if recorder is True else recorder,
    )


def events(committed, action_id):
    with committed() as s:
        return list(
            s.scalars(
                select(AuditEvent)
                .where(AuditEvent.action_request_id == action_id)
                .order_by(AuditEvent.created_at, AuditEvent.event_type)
            )
        )


def runs(committed):
    with committed() as s:
        return list(s.scalars(select(AgentRun).order_by(AgentRun.created_at)))


PROPOSE = ai_tools(
    call(PROPOSE_CANCEL_ORDER, "tc-1", order_number="ORD-1004", reason="PRIVATE-REASON")
)


def test_action_lifecycle_is_audited_with_correlation_ids(committed, svc, ns):
    a = agent(committed, svc, PROPOSE)
    pending = a.run("PRIVATE-USER-TEXT cancel ORD-1004", ns, thread_id="t").action
    aid = uuid.UUID(pending.id)
    a.resume(ns, thread_id="t", action_id=aid, decision="approve", decided_by="user-42")
    svc.execute(ns.tenant, aid)  # duplicate execution attempt
    got = [(e.event_type, e.actor) for e in events(committed, aid)]
    assert got[0] == ("action_requested", "agent")
    assert ("approval_decided", "user-42") in got and ("action_succeeded", "system") in got
    assert got[-1] == ("action_duplicate_prevented", "system")
    requested = events(committed, aid)[0]
    assert (requested.request_id, requested.tool_call_id) == ("req-obs-1", "tc-1")
    for e in events(committed, aid):
        assert set(e.details) <= SAFE_DETAIL_KEYS
        assert "PRIVATE" not in json.dumps(e.details)


@pytest.mark.parametrize("decision", ["reject", "expire"])
def test_rejection_and_expiry_are_audited(committed, svc, ns, clock, decision):
    a = agent(committed, svc, PROPOSE)
    aid = uuid.UUID(a.run("cancel ORD-1004", ns, thread_id="t").action.id)
    if decision == "expire":
        clock.now = T0 + timedelta(days=1)
    a.resume(ns, thread_id="t", action_id=aid, decision="reject", decided_by="u")
    types = [e.event_type for e in events(committed, aid)]
    assert types == [
        "action_requested",
        "approval_expired" if decision == "expire" else "approval_decided",
    ]


def test_refused_decision_leaves_no_audit_event(committed, svc, ns):
    a = agent(committed, svc, PROPOSE)
    aid = uuid.UUID(a.run("cancel ORD-1004", ns, thread_id="t").action.id)
    with pytest.raises(E.ArgumentsHashMismatchError):
        a.resume(
            ns,
            thread_id="t",
            action_id=aid,
            decision="approve",
            decided_by="u",
            expected_hash="0" * 64,
        )
    assert [e.event_type for e in events(committed, aid)] == ["action_requested"]


def test_failed_execution_is_audited(committed, svc, ns, monkeypatch):
    from sqlalchemy.exc import OperationalError

    a = agent(committed, svc, PROPOSE)
    aid = uuid.UUID(a.run("cancel ORD-1004", ns, thread_id="t").action.id)
    monkeypatch.setattr(
        svc, "_apply", lambda *_: (_ for _ in ()).throw(OperationalError("x", {}, Exception("y")))
    )
    a.resume(ns, thread_id="t", action_id=aid, decision="approve", decided_by="u")
    failed = [e for e in events(committed, aid) if e.event_type == "action_failed"]
    assert len(failed) == 1 and failed[0].details["failure_code"] == "execution_error"


def test_runs_are_recorded_without_content(committed, svc, ns):
    a = agent(committed, svc, ai_tools(call("get_order", order_number="ORD-1004")), PROPOSE)
    pending = a.run("PRIVATE-USER-TEXT", ns, thread_id="t").action
    a.resume(ns, thread_id="t", action_id=uuid.UUID(pending.id), decision="approve", decided_by="u")
    run, resume = runs(committed)
    assert (run.kind, run.outcome, run.model_calls, run.commerce_tool_count, run.action_status) == (
        "run",
        "approval_pending",
        2,
        1,
        "pending_approval",
    )
    assert (resume.kind, resume.outcome, resume.action_status, resume.model_calls) == (
        "resume",
        "ok",
        "succeeded",
        2,
    )
    assert run.request_id == "req-obs-1" and run.thread_key.startswith("cg1-")
    assert run.action_request_id == resume.action_request_id == uuid.UUID(pending.id)
    assert (run.profile, run.prompt_version) == ("agent", "commerce-assistant-v4")
    with committed() as s:
        raw = json.dumps([list(map(str, r)) for r in s.execute(select(AgentRun.__table__)).all()])
    assert "PRIVATE" not in raw and str(ns.tenant_id) in raw  # tenant column only


def test_grounding_failures_are_countable(committed, svc, ns):
    from app.agent.assistant import AssistantError

    # An unretrieved citation without any retrieval gets ONE corrective call; the model
    # repeats it, so the run still fails closed and is counted as a grounding failure.
    bad = "See [policy://made-up/v1#chunk-1]"
    a = agent(committed, svc, ai_text(bad), ai_text(bad))
    with pytest.raises(AssistantError):
        a.run("policy?", ns, thread_id="t")
    [run] = runs(committed)
    assert (run.outcome, run.error_detail, run.grounding_failure) == (
        "agent_grounding_error",
        "citation_not_retrieved",
        True,
    )


def test_run_recorder_failure_never_breaks_the_run(committed, svc, ns, caplog):
    @contextmanager
    def broken():
        raise RuntimeError("audit db down")
        yield  # pragma: no cover

    a = agent(committed, svc, ai_text("Hello."), recorder=RunRecorder(broken))
    with caplog.at_level(logging.WARNING):
        assert a.run("hi", ns, thread_id="t").answer == "Hello."
    assert any(r.message == "agent run not recorded" for r in caplog.records)


def test_langsmith_tracing_is_off_by_default_and_needs_a_key(monkeypatch):
    from app.observability.tracing import configure_tracing

    monkeypatch.setenv("LANGSMITH_TRACING", "true")  # ambient env must not switch it on
    assert Settings(_env_file=None).langsmith_tracing is True  # env maps to the setting...
    assert configure_tracing(Settings(_env_file=None, langsmith_tracing=True)) is False  # no key
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert configure_tracing(Settings(_env_file=None, langsmith_tracing=False)) is False
    monkeypatch.delenv("LANGSMITH_TRACING")
    assert Settings(_env_file=None).langsmith_tracing is False
