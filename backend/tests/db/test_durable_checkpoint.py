"""Durable PostgreSQL checkpoints (official langgraph-checkpoint-postgres) for the approval
flow: pauses survive "process restarts" (a new pool + saver + assistant), can be resumed
from another OS process, stay tenant-isolated, and a crash stays distinguishable from an
intentional approval interrupt."""

import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select, text

from app.actions.capability import PROPOSE_CANCEL_ORDER
from app.actions.service import ActionService
from app.agent.assistant import AssistantError, AssistantLimits
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.checkpoint import delete_checkpoint_thread, durable_checkpointer
from app.agent.graph.profile import AGENT_PROFILE
from app.agent.graph.state import is_failure_marker
from app.models import Order
from tests.assistant.fakes import ai_text, ai_tools, call, make_provider
from tests.rag.fakes import FakeRetriever

BACKEND = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
CHECKPOINT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def durable(database_url, db_engine):
    """Factory of fresh DurableCheckpointers (= fresh 'processes'); tables wiped after."""
    opened = []

    def make():
        cp = durable_checkpointer(database_url, min_size=1, max_size=2)
        cp.setup()
        opened.append(cp)
        return cp

    try:
        yield make
    finally:
        for cp in opened:
            cp.close()
        with db_engine.begin() as conn:  # not Alembic-managed: drop so migrations stay exact
            for table in (*CHECKPOINT_TABLES, "checkpoint_migrations"):
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))


@pytest.fixture
def ns(tenant_a):
    return AgentContext(tenant_a.tenant_id, "req-ns")


@pytest.fixture
def bp(tenant_b):
    return AgentContext(tenant_b.tenant_id, "req-bp")


def process(committed, clock, saver, *script):
    """One 'process': its own assistant + action service on the shared durable saver."""
    provider, model = make_provider(*script)
    assistant = CommerceGraphAssistant(
        provider,
        limits=AssistantLimits(),
        checkpointer=saver,
        retriever=FakeRetriever(),
        profile=AGENT_PROFILE,
        actions=ActionService(committed, clock=clock),
    )
    return assistant, model


def propose(number="ORD-1004"):
    return ai_tools(
        call(PROPOSE_CANCEL_ORDER, "p1", order_number=number, reason="Customer request")
    )


def status(committed, ctx, number="ORD-1004"):
    with committed() as s:
        return s.scalar(
            select(Order.status).where(
                Order.tenant_id == ctx.tenant_id, Order.order_number == number
            )
        )


def pause_in_first_process(committed, clock, durable, ctx):
    cp = durable()
    a, _ = process(committed, clock, cp.saver, propose())
    pending = a.run("Please cancel ORD-1004", ctx, thread_id="t").action
    cp.close()  # the first process is gone
    return pending


def test_approval_survives_restart_and_resumes_in_a_new_process(
    committed, clock, durable, ns, caplog
):
    pending = pause_in_first_process(committed, clock, durable, ns)
    b, model = process(committed, clock, durable().saver)
    assert b.pending_approval(ns, "t")["id"] == pending.id
    done = b.resume(
        ns,
        thread_id="t",
        action_id=uuid.UUID(pending.id),
        decision="approve",
        decided_by="user-1",
        expected_hash=pending.arguments_hash,
    )
    assert done.action.status == "succeeded" and model.invocations == []
    assert status(committed, ns) == "cancelled"
    assert b.pending_approval(ns, "t") is None
    # strict msgpack: graph state holds only safe types (no enum/class revival warnings)
    assert not [r for r in caplog.records if r.name.startswith("langgraph.checkpoint.serde")]


def test_reject_after_restart(committed, clock, durable, ns):
    pending = pause_in_first_process(committed, clock, durable, ns)
    b, _ = process(committed, clock, durable().saver)
    res = b.resume(
        ns, thread_id="t", action_id=uuid.UUID(pending.id), decision="reject", decided_by="u"
    )
    assert res.action.status == "rejected" and status(committed, ns) == "processing"


def test_expired_approval_after_restart(committed, clock, durable, ns):
    pending = pause_in_first_process(committed, clock, durable, ns)
    clock.now = T0 + timedelta(hours=2)
    b, _ = process(committed, clock, durable().saver)
    res = b.resume(
        ns, thread_id="t", action_id=uuid.UUID(pending.id), decision="approve", decided_by="u"
    )
    assert res.action.status == "expired" and status(committed, ns) == "processing"


def test_resume_from_another_os_process(committed, clock, durable, ns, database_url):
    clock.now = datetime.now(UTC)  # the other process uses the real clock
    pending = pause_in_first_process(committed, clock, durable, ns)
    script = f"""
import uuid, json
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.checkpoint import durable_checkpointer
from app.agent.graph.profile import AGENT_PROFILE
from app.agent.context import AgentContext
from app.actions.service import ActionService
from tests.assistant.fakes import make_provider
cp = durable_checkpointer({database_url!r})
provider, _ = make_provider()
a = CommerceGraphAssistant(
    provider, checkpointer=cp.saver, profile=AGENT_PROFILE, actions=ActionService(),
    retriever=lambda: None,
)
ctx = AgentContext(uuid.UUID({str(ns.tenant_id)!r}))
res = a.resume(
    ctx, thread_id="t", action_id=uuid.UUID({pending.id!r}), decision="approve",
    decided_by="proc-2",
)
print(json.dumps({{"status": res.action.status}}))
cp.close()
"""
    env = {**os.environ, "DATABASE_URL": database_url, "APP_ENV": "test"}
    env.pop("TEST_DATABASE_URL", None)
    out = subprocess.run(  # noqa: S603 - fixed interpreter and test-authored script
        [sys.executable, "-c", script],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"status": "succeeded"}
    assert status(committed, ns) == "cancelled"


def test_same_raw_thread_id_in_two_tenants_is_isolated(
    committed, clock, durable, ns, bp, db_engine
):
    pause_in_first_process(committed, clock, durable, ns)
    b, _ = process(committed, clock, durable().saver, ai_text("Hello BluePeak."))
    assert b.pending_approval(bp, "t") is None  # a different, empty thread
    assert b.run("Hi", bp, thread_id="t").answer == "Hello BluePeak."
    with pytest.raises(AssistantError) as exc:  # Northstar's thread is still paused
        b.run("Hi", ns, thread_id="t")
    assert exc.value.code == "agent_approval_pending"
    with db_engine.connect() as conn:
        keys = set(conn.execute(text("SELECT DISTINCT thread_id FROM checkpoints")).scalars())
        blobs = b"".join(
            bytes(r[0])
            for r in conn.execute(text("SELECT blob FROM checkpoint_blobs WHERE blob IS NOT NULL"))
        )
        raw = json.dumps(
            [list(map(str, r)) for r in conn.execute(text("SELECT * FROM checkpoints"))]
        )
    assert keys == {
        b.thread_config(ns, "t")["configurable"]["thread_id"],
        b.thread_config(bp, "t")["configurable"]["thread_id"],
    }
    assert all(k.startswith("cg1-") and k != "t" for k in keys)
    for tenant in (ns.tenant_id, bp.tenant_id):  # runtime context is never checkpointed
        assert str(tenant).encode() not in blobs and str(tenant) not in raw


def test_other_tenant_cannot_resume_through_the_public_runner(committed, clock, durable, ns, bp):
    pending = pause_in_first_process(committed, clock, durable, ns)
    b, _ = process(committed, clock, durable().saver)
    with pytest.raises(AssistantError) as exc:
        b.resume(
            bp, thread_id="t", action_id=uuid.UUID(pending.id), decision="approve", decided_by="u"
        )
    assert exc.value.code == "agent_no_pending_approval"
    assert status(committed, ns) == "processing"


def test_crash_is_distinguished_from_an_intentional_interrupt(committed, clock, durable, ns):
    cp = durable()
    crashing, _ = process(
        committed,
        clock,
        cp.saver,
        ai_tools(call("get_order", "r1", order_number="ORD-1004")),
        SystemExit("process killed"),
    )
    with pytest.raises(SystemExit):  # dies mid-run: no failure marker, no interrupt recorded
        crashing.run("Where is ORD-1004?", ns, thread_id="crash")
    cp.close()
    b, _ = process(committed, clock, durable().saver, ai_text("It is processing."))
    assert b.pending_approval(ns, "crash") is None  # not mistaken for an approval pause
    assert b.run("Try again", ns, thread_id="crash").answer == "It is processing."
    msgs = b.graph.get_state(b.thread_config(ns, "crash")).values["messages"]
    assert sum(is_failure_marker(m) for m in msgs) == 1  # crashed turn closed on resume


def test_thread_retention_delete(committed, clock, durable, ns, db_engine):
    cp = durable()
    a, _ = process(committed, clock, cp.saver, ai_text("hello"))
    a.run("hi", ns, thread_id="gone")
    key = a.thread_config(ns, "gone")["configurable"]["thread_id"]
    delete_checkpoint_thread(cp.saver, key)
    with db_engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM checkpoints WHERE thread_id=:k"), {"k": key}
            ).scalar()
            == 0
        )


def test_durable_saver_never_uses_pickle(durable):
    cp = durable()
    assert cp.saver.serde.pickle_fallback is False
    assert cp.pool.kwargs["autocommit"] is True and cp.pool.kwargs["prepare_threshold"] == 0
