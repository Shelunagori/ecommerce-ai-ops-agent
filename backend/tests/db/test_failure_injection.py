"""Deterministic failure injection (Step 10, Phase 4). Each test names ONE failure and its
EXPECTED behaviour. Global rules: fail closed, never fabricate a successful write, never turn
an infrastructure failure into "no policy", never auto-repeat an uncertain side effect, stable
codes, sanitised messages.

| Failure                                   | Expected                                           |
| ----------------------------------------- | -------------------------------------------------- |
| DB unavailable (commerce read)            | tool envelope service_unavailable, run continues   |
| DB timeout (commerce read)                | tool envelope service_unavailable                  |
| DB unavailable (action proposal)          | agent_action_error/database_unavailable, no request|
| LLM timeout                               | llm_timeout (after the bounded retry)              |
| embedding timeout / profile missing       | agent_retrieval_error/<embedding code>, no answer  |
| unexpected tool exception                 | tool envelope internal_error, no traceback         |
| retrieval exception                       | agent_retrieval_error/retrieval_failed             |
| malformed tool call                       | agent_protocol_error/invalid_tool_calls            |
| malformed structured output               | llm_output_invalid                                 |
| duplicate model call id                   | agent_protocol_error/tool_call_id                  |
| duplicate idempotency key, other args     | idempotency_conflict, nothing stored               |
| action DB failure after approval          | action failed/execution_error, no write            |
| precondition changed before execution     | action failed/order_not_cancellable, no write      |
| process restart during approval           | resumable (test_durable_checkpoint.py)             |
| expired approval                          | expired, no write                                  |
| repeated resume                           | agent_no_pending_approval, single write            |
| no retrieval result                       | no_results ToolMessage, answer without citations   |
| checkpoint persistence error              | agent_state_unavailable; approved action idempotent|
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from psycopg.errors import QueryCanceled
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

import app.db.session as db_session_module
from app.actions import errors as E  # noqa: N812
from app.actions.capability import PROPOSE_CANCEL_ORDER, PROPOSE_STORE_CREDIT
from app.actions.service import ActionService
from app.agent.assistant import AssistantError, AssistantLimits
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.profile import AGENT_PROFILE
from app.agent.llm.errors import LLMError
from app.agent.tools import ToolDependencies, build_commerce_tools
from app.knowledge.embeddings.errors import (
    EmbeddingProfileNotMaterializedError,
    EmbeddingTimeoutError,
)
from app.models import ActionRequest, Order, StoreCreditTransaction
from tests.assistant.fakes import ai_text, ai_tools, call, make_provider
from tests.db.conftest import fixed_clock
from tests.rag.fakes import FakeRetriever, result, search

T0 = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
REQ = httpx.Request("POST", "http://localhost:11434/api/chat")


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now


class FlakySaver(InMemorySaver):
    """InMemorySaver whose writes can be made to fail (checkpoint persistence error)."""

    fail = False

    def put(self, *a: Any, **k: Any):
        if self.fail:
            raise OperationalError("INSERT INTO checkpoints", {}, Exception("disk full"))
        return super().put(*a, **k)

    def put_writes(self, *a: Any, **k: Any):
        if self.fail:
            raise OperationalError("INSERT INTO checkpoint_writes", {}, Exception("disk full"))
        return super().put_writes(*a, **k)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def svc(committed, clock):
    return ActionService(committed, clock=clock)


@pytest.fixture
def tools(committed):
    return build_commerce_tools(
        ToolDependencies(session_scope=db_session_module.read_only_session, clock=fixed_clock)
    )


@pytest.fixture
def ns(tenant_a):
    return AgentContext(tenant_a.tenant_id, "req-fi")


def agent(tools, svc, *script, saver=None, retriever=None, retries=1):
    provider, model = make_provider(*script, retries=retries)
    return CommerceGraphAssistant(
        provider,
        tools=tools,
        limits=AssistantLimits(),
        checkpointer=saver or InMemorySaver(),
        retriever=retriever or FakeRetriever(),
        profile=AGENT_PROFILE,
        actions=svc,
    ), model


def cancel_call(number="ORD-1004", cid="p1"):
    return ai_tools(call(PROPOSE_CANCEL_ORDER, cid, order_number=number, reason="Customer request"))


def count(committed, model):
    with committed() as s:
        return s.scalar(select(func.count()).select_from(model))


def order_status(committed, ctx, number="ORD-1004"):
    with committed() as s:
        return s.scalar(
            select(Order.status).where(
                Order.tenant_id == ctx.tenant_id, Order.order_number == number
            )
        )


def failing_scope(exc):
    from contextlib import contextmanager

    @contextmanager
    def scope():
        raise exc
        yield  # pragma: no cover

    return scope


# --- database -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc",
    [
        OperationalError("SELECT", {}, Exception("down")),
        OperationalError("SELECT", {}, QueryCanceled("timeout")),
    ],
    ids=["unavailable", "timeout"],
)
def test_db_failure_on_commerce_read_is_service_unavailable(svc, ns, exc):
    tools = build_commerce_tools(
        ToolDependencies(session_scope=failing_scope(exc), clock=fixed_clock)
    )
    a, _ = agent(
        tools,
        svc,
        ai_tools(call("get_order", order_number="ORD-1004")),
        ai_text("Unavailable right now."),
    )
    res = a.run("ORD-1004?", ns, thread_id="t")
    assert res.tool_calls[0].outcome == "service_unavailable" and res.action is None


def test_db_failure_on_action_proposal(tools, committed, ns):
    broken = ActionService(failing_scope(OperationalError("INSERT", {}, Exception("down"))))
    a, _ = agent(tools, broken, cancel_call())
    with pytest.raises(AssistantError) as exc:
        a.run("Cancel ORD-1004", ns, thread_id="t")
    assert (exc.value.code, exc.value.detail) == ("agent_action_error", "database_unavailable")
    assert "down" not in exc.value.message and count(committed, ActionRequest) == 0


# --- models / embeddings / retrieval ---------------------------------------------------
def test_llm_timeout(tools, svc, ns):
    a, model = agent(
        tools, svc, httpx.ReadTimeout("slow", request=REQ), httpx.ReadTimeout("slow", request=REQ)
    )
    with pytest.raises(AssistantError) as exc:
        a.run("hi", ns, thread_id="t")
    assert exc.value.code == "llm_timeout" and len(model.invocations) == 2  # one bounded retry


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (EmbeddingTimeoutError(), "embedding_timeout"),
        (EmbeddingProfileNotMaterializedError(), "embedding_profile_not_materialized"),
        (RuntimeError("retriever bug"), "retrieval_failed"),
        (
            OperationalError("SELECT", {}, QueryCanceled("statement timeout")),
            "database_unavailable",
        ),
    ],
)
def test_retrieval_side_failures_are_never_no_policy(tools, svc, ns, error, detail):
    a, model = agent(
        tools,
        svc,
        ai_tools(search()),
        ai_text("From memory: 15%."),
        retriever=FakeRetriever([error]),
    )
    with pytest.raises(AssistantError) as exc:
        a.run("What compensation applies?", ns, thread_id="t")
    assert (exc.value.code, exc.value.detail) == ("agent_retrieval_error", detail)
    assert len(model.invocations) == 1  # no answer from memory


def test_unexpected_tool_exception_is_internal_error(svc, ns):
    tools = build_commerce_tools(
        ToolDependencies(session_scope=failing_scope(ZeroDivisionError("bug")), clock=fixed_clock)
    )
    a, model = agent(
        tools, svc, ai_tools(call("get_order", order_number="ORD-1004")), ai_text("Sorry.")
    )
    res = a.run("ORD-1004?", ns, thread_id="t")
    assert res.tool_calls[0].outcome == "internal_error"
    sent = " ".join(str(m.content) for m in model.invocations[1].messages)
    assert "ZeroDivisionError" not in sent and "bug" not in sent


def test_malformed_tool_call(tools, svc, ns):
    bad = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "id": "x",
                "name": PROPOSE_CANCEL_ORDER,
                "args": "{",
                "error": "e",
            }
        ],
    )
    a, _ = agent(tools, svc, bad)
    with pytest.raises(AssistantError) as exc:
        a.run("Cancel", ns, thread_id="t")
    assert (exc.value.code, exc.value.detail) == ("agent_protocol_error", "invalid_tool_calls")


def test_malformed_structured_output():
    from app.agent.llm.provider import ChatModelProvider, ProviderInfo, RetryPolicy

    class Schema(BaseModel):
        intent: str

    class BrokenStructured:
        def with_structured_output(self, *_a, **_k):
            return self

        def invoke(self, _messages):
            return {"not": "the schema"}

    provider = ChatModelProvider(
        BrokenStructured(),
        ProviderInfo("fake", "m"),
        RetryPolicy(max_retries=0),
        sleep=lambda _s: None,
    )
    with pytest.raises(LLMError) as exc:
        provider.invoke_structured(Schema, [], operation="t")
    assert exc.value.code == "llm_output_invalid"


def test_duplicate_model_call_id(tools, svc, ns):
    a, _ = agent(
        tools,
        svc,
        ai_tools(
            call("get_order", "dup", order_number="A"), call("get_order", "dup", order_number="B")
        ),
    )
    with pytest.raises(AssistantError) as exc:
        a.run("x", ns, thread_id="t")
    assert (exc.value.code, exc.value.detail) == ("agent_protocol_error", "tool_call_id")


def test_no_retrieval_result(tools, svc, ns):
    a, _ = agent(
        tools,
        svc,
        ai_tools(search()),
        ai_text("No applicable policy was found."),
        retriever=FakeRetriever([result()]),
    )
    res = a.run("Policy?", ns, thread_id="t")
    assert res.retrievals[0].outcome == "no_results" and res.citations == []


# --- actions ------------------------------------------------------------------------------
def test_duplicate_idempotency_key_with_other_arguments(svc, committed, ns):
    base = {"evidence": [], "requested_by": {"runner": "t"}, "idempotency_key": "idk-fixed"}
    svc.propose(
        ns.tenant,
        "cancel_order",
        {"order_number": "ORD-1004", "reason": "a"},
        thread_key=None,
        tool_call_id=None,
        **base,
    )
    with pytest.raises(E.IdempotencyConflictError):
        svc.propose(
            ns.tenant,
            "cancel_order",
            {"order_number": "ORD-1007", "reason": "a"},
            thread_key=None,
            tool_call_id=None,
            **base,
        )
    assert count(committed, ActionRequest) == 1


def test_action_db_failure_after_approval(tools, svc, committed, ns, monkeypatch):
    a, _ = agent(tools, svc, cancel_call())
    pending = a.run("Cancel ORD-1004", ns, thread_id="t").action
    monkeypatch.setattr(
        svc,
        "_apply",
        lambda *_: (_ for _ in ()).throw(OperationalError("UPDATE", {}, Exception("down"))),
    )
    res = a.resume(
        ns, thread_id="t", action_id=uuid.UUID(pending.id), decision="approve", decided_by="u"
    )
    assert (res.action.status, res.action.failure_code) == ("failed", "execution_error")
    assert order_status(committed, ns) == "processing"


def test_expired_approval(tools, svc, committed, ns, clock):
    a, _ = agent(tools, svc, cancel_call())
    pending = a.run("Cancel ORD-1004", ns, thread_id="t").action
    clock.now = T0 + timedelta(days=1)
    res = a.resume(
        ns, thread_id="t", action_id=uuid.UUID(pending.id), decision="approve", decided_by="u"
    )
    assert res.action.status == "expired" and order_status(committed, ns) == "processing"


def test_repeated_resume_writes_once(tools, svc, committed, ns):
    credit = ai_tools(
        call(
            PROPOSE_STORE_CREDIT,
            "c1",
            customer_code="CUS-1002",
            amount="15.00",
            currency="USD",
            reason="Delay",
            order_number="ORD-1003",
            policy_citations=["policy://delayed-shipment-compensation/v2#chunk-2"],
        )
    )
    a, _ = agent(tools, svc, ai_tools(search()), credit)
    pending = a.run("Credit CUS-1002", ns, thread_id="t").action
    aid = uuid.UUID(pending.id)
    a.resume(ns, thread_id="t", action_id=aid, decision="approve", decided_by="u")
    for _ in range(3):
        with pytest.raises(AssistantError) as exc:
            a.resume(ns, thread_id="t", action_id=aid, decision="approve", decided_by="u")
        assert exc.value.code == "agent_no_pending_approval"
    assert count(committed, StoreCreditTransaction) == 1


def test_checkpoint_persistence_error_is_stable_and_retry_is_idempotent(tools, svc, committed, ns):
    saver = FlakySaver()
    a, _ = agent(tools, svc, cancel_call(), saver=saver)
    pending = a.run("Cancel ORD-1004", ns, thread_id="t").action
    saver.fail = True
    with pytest.raises(AssistantError) as exc:
        a.resume(
            ns, thread_id="t", action_id=uuid.UUID(pending.id), decision="approve", decided_by="u"
        )
    assert exc.value.code == "agent_state_unavailable" and "disk" not in exc.value.message
    saver.fail = False
    # The graph still sits at the approval pause; resuming again completes without a 2nd write.
    res = a.resume(
        ns, thread_id="t", action_id=uuid.UUID(pending.id), decision="approve", decided_by="u"
    )
    assert res.action.status == "succeeded" and order_status(committed, ns) == "cancelled"


def test_checkpoint_read_error_on_a_new_message(tools, svc, ns):
    saver = FlakySaver()
    a, _ = agent(tools, svc, ai_text("hello"), saver=saver)
    saver.fail = True
    with pytest.raises(AssistantError) as exc:
        a.run("hi", ns, thread_id="t")
    assert exc.value.code == "agent_state_unavailable"
