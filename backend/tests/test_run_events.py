"""RunEmitter and the SSE transport (no database): ordering, isolation from sink failures,
heartbeats and disconnect behaviour."""

from __future__ import annotations

import asyncio
import threading

from pydantic import BaseModel

from app.agent.events import (
    NULL_EMITTER,
    Capability,
    RunEmitter,
    current_emitter,
    emitter_scope,
)
from app.agent.trace import TraceEventKind, TraceStatus, TraceStep, tool_step
from app.api.streaming import encode_event, stream_run


def collect():
    events = []
    return events, RunEmitter(events.append)


def test_events_are_sequenced_and_steps_pair_up():
    events, em = collect()
    em.run_started([Capability(kind=TraceEventKind.RESPONSE, label="Response generated")])
    step = em.start(TraceEventKind.COMMERCE_TOOL, "Tool: get_order", "lookup", tool="get_order")
    em.finish(step, tool_step({"tool": "get_order", "outcome": "success", "duration_ms": 3.0}))
    assert [e.sequence for e in events] == [1, 2, 3]
    assert [e.type for e in events] == ["run_started", "step_started", "step_completed"]
    assert events[1].step_id == events[2].step_id == step
    assert events[1].status == "running" and events[2].status == "completed"
    assert events[2].duration_ms is not None  # measured by the emitter


def test_failed_trace_step_becomes_step_failed():
    events, em = collect()
    step = em.start(TraceEventKind.RETRIEVAL, "Policy retrieval")
    em.finish(step, TraceStep.of(TraceEventKind.RETRIEVAL, "Policy retrieval", TraceStatus.FAILED))
    assert (events[-1].type, events[-1].status) == ("step_failed", "failed")


def test_open_steps_are_failed_never_completed():
    events, em = collect()
    em.start(TraceEventKind.MODEL, "Agent orchestration")
    em.fail_open_steps("The step did not complete")
    assert (events[-1].type, events[-1].status, events[-1].detail) == (
        "step_failed",
        "failed",
        "The step did not complete",
    )
    em.fail_open_steps("again")  # nothing left open
    assert len(events) == 2


def test_skip_unused_only_skips_kinds_that_never_ran():
    events, em = collect()
    caps = [
        Capability(kind=TraceEventKind.COMMERCE_TOOL, label="Commerce tools"),
        Capability(kind=TraceEventKind.RETRIEVAL, label="Policy retrieval"),
    ]
    step = em.start(TraceEventKind.COMMERCE_TOOL, "Tool: get_order")
    em.finish(step, tool_step({"tool": "get_order", "outcome": "success", "duration_ms": 1.0}))
    em.skip_unused(caps, "Not used in this run")
    assert [(e.type, e.kind) for e in events[-1:]] == [("step_skipped", "retrieval")]


def test_a_failing_sink_never_raises_into_the_run():
    def boom(_e):
        raise RuntimeError("socket closed")

    em = RunEmitter(boom)
    em.run_started([])
    step = em.start(TraceEventKind.MODEL, "Agent orchestration")
    em.finish(step, TraceStep.of(TraceEventKind.MODEL, "Agent orchestration"))
    em.skip(TraceEventKind.RETRIEVAL, "Policy retrieval", "Not used")
    em.run_failed("x", "y", status=500)  # all silently dropped


def test_scope_installs_and_restores_the_emitter_per_context():
    _, em = collect()
    assert current_emitter() is NULL_EMITTER
    seen = {}
    with emitter_scope(em):
        assert current_emitter() is em
        t = threading.Thread(target=lambda: seen.setdefault("other", current_emitter()))
        t.start()
        t.join()
    assert current_emitter() is NULL_EMITTER
    assert seen["other"] is NULL_EMITTER  # a foreign thread never inherits another run's sink


def test_null_emitter_emits_nothing_and_keeps_no_state():
    assert NULL_EMITTER.start(TraceEventKind.MODEL, "x") is None
    NULL_EMITTER.fail_open_steps("x")
    assert NULL_EMITTER.kinds_seen == set()


def test_sse_encoding():
    events, em = collect()
    em.run_failed("rate_limited", "Too many requests.", status=429)
    raw = encode_event(events[0]).decode()
    assert raw.startswith("event: run_failed\nid: 1\ndata: {") and raw.endswith("}\n\n")
    assert '"status":429' in raw and "null" not in raw  # None fields omitted


class Out(BaseModel):
    ok: bool


def _drain(response, *, stop_after: int | None = None):
    async def run():
        chunks = []
        it = response.body_iterator
        async for chunk in it:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
            if stop_after is not None and len(chunks) >= stop_after:
                await it.aclose()  # the client disconnected
                break
        return chunks

    return run


def test_stream_run_ends_with_run_completed_and_heartbeats_while_idle():
    gate = threading.Event()

    def work(em):
        step = em.start(TraceEventKind.MODEL, "Agent orchestration")
        gate.wait(1.0)
        em.finish(step, TraceStep.of(TraceEventKind.MODEL, "Agent orchestration"))
        return Out(ok=True)

    async def main():
        response = stream_run(work, heartbeat_seconds=0.05)
        task = asyncio.create_task(_drain(response)())
        await asyncio.sleep(0.2)  # idle while "the model" works -> heartbeats
        gate.set()
        return b"".join(await task).decode()

    raw = asyncio.run(main())
    assert ": keep-alive" in raw
    assert raw.rstrip().split("\n\n")[-1].startswith("event: run_completed")
    assert '"response":{"ok":true}' in raw


def test_disconnect_does_not_cancel_the_run():
    gate, done = threading.Event(), threading.Event()

    def work(em):
        em.start(TraceEventKind.ACTION_EXECUTION, "Deterministic execution")
        gate.wait(2.0)
        done.set()  # e.g. the transaction commits
        return Out(ok=True)

    async def main():
        response = stream_run(work)
        chunks = await _drain(response, stop_after=1)()  # client leaves after the first event
        gate.set()
        for _ in range(100):
            if done.is_set():
                break
            await asyncio.sleep(0.02)
        return chunks

    chunks = asyncio.run(main())
    assert len(chunks) == 1 and done.is_set()


def test_worker_exceptions_become_a_safe_run_failed():
    def work(_em):
        raise ValueError("boom at postgresql://user:pw@db/prod")

    async def main():
        return b"".join(await _drain(stream_run(work))()).decode()

    raw = asyncio.run(main())
    assert "event: run_failed" in raw and '"code":"internal_error"' in raw
    assert "postgresql" not in raw and "boom" not in raw
