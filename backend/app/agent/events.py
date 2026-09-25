"""Live run events: the agent's REAL execution boundaries, streamed while a run executes.

    emitter = RunEmitter(sink)          # sink: Callable[[RunEvent], None], e.g. a queue
    assistant.run(text, context, thread_id=..., events=emitter)

The runner installs the emitter for the duration of ONE run (a context variable, inherited
by the threads LangGraph runs nodes in); graph nodes then report the boundaries they actually
cross - model call before/after, each commerce tool, the policy retrieval, grounding
validation, action proposal, the approval pause/decision and deterministic execution. Nothing
is emitted on a timer and nothing is predicted: a step is ``running`` only between the real
start and end of that operation.

Finished steps are built with the SAME builders as the final ``execution_trace``
(``app.agent.trace``), so a streamed completed step and its final-trace entry cannot disagree.

Safety and isolation:
* Events carry only fixed labels, safe details and the closed metadata set of the trace
  (identifiers, counts, outcomes, measured durations). Never prompts, messages, model output
  or reasoning, tool arguments, retrieved text, SQL, embeddings, tenant ids, tokens or URLs.
* Emitting can NEVER change a run: every sink call is isolated (exceptions are swallowed and
  logged once), events are produced after the step's own work, and nothing here touches
  graph state, tenant scope, the database or the action service.
* Without an emitter (the non-streaming API, the CLI, tests) nodes use the no-op emitter.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.trace import MetaValue, TraceEventKind, TraceStatus, TraceStep

logger = logging.getLogger("app.agent.events")


class RunEventType(StrEnum):
    RUN_STARTED = "run_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_SKIPPED = "step_skipped"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_RESOLVED = "approval_resolved"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class StepStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"  # refused by a rule or by the human (not an error)
    WAITING = "waiting"  # waiting for a human decision (not running)
    SKIPPED = "skipped"  # the run finished (or stopped) without this capability


class Capability(BaseModel):
    """A step kind this run's graph CAN execute (not a prediction that it will)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: TraceEventKind
    label: str = Field(max_length=80)


class RunError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    code: str = Field(max_length=64)
    message: str = Field(max_length=300)
    status: int | None = Field(default=None, ge=400, le=599)  # the JSON API's HTTP status


class RunEvent(BaseModel):
    """One streamed event. ``sequence`` orders events within a run (1, 2, ...);
    ``step_id`` ties a step's start to its end; ``elapsed_ms`` is server time since the run
    started (no wall clock); ``duration_ms`` is the step's MEASURED duration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: RunEventType
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    sequence: int = Field(ge=1)
    elapsed_ms: float = Field(ge=0)
    step_id: str | None = Field(default=None, pattern=r"^s[0-9]{1,4}$")
    kind: TraceEventKind | None = None
    label: str | None = Field(default=None, max_length=80)
    status: StepStatus | None = None
    detail: str | None = Field(default=None, max_length=160)
    metadata: dict[str, MetaValue] = {}
    duration_ms: float | None = Field(default=None, ge=0)
    capabilities: list[Capability] | None = None  # run_started
    next_steps: list[Capability] | None = None  # approval_required: fixed graph topology
    response: dict[str, Any] | None = None  # run_completed: the normal API response body
    error: RunError | None = None  # run_failed


_TRACE_TO_STEP = {
    TraceStatus.COMPLETED: StepStatus.COMPLETED,
    TraceStatus.FAILED: StepStatus.FAILED,
    TraceStatus.REJECTED: StepStatus.REJECTED,
    TraceStatus.WAITING: StepStatus.WAITING,
}


class RunEmitter:
    """Builds ordered, validated events for ONE run and hands them to ``sink``.

    Thread-safe (one lock); every public method is exception-proof by contract."""

    def __init__(self, sink: Callable[[RunEvent], None], *, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self._sink = sink
        self._lock = threading.Lock()
        self._sequence = 0
        self._steps = 0
        self._open: dict[str, tuple[float, TraceEventKind, str]] = {}
        self._clock = time.perf_counter
        self._started = self._clock()
        self._warned = False
        self.kinds_seen: set[TraceEventKind] = set()

    # --- lifecycle -----------------------------------------------------------------------------
    def run_started(self, capabilities: list[Capability]) -> None:
        self._emit(RunEventType.RUN_STARTED, capabilities=capabilities)

    def run_completed(self, response: dict[str, Any]) -> None:
        self._emit(RunEventType.RUN_COMPLETED, response=response)

    def run_failed(self, code: str, message: str, *, status: int | None = None) -> None:
        self._emit(
            RunEventType.RUN_FAILED, error=RunError(code=code, message=message, status=status)
        )

    # --- steps ---------------------------------------------------------------------------------
    def start(
        self, kind: TraceEventKind, label: str, detail: str | None = None, **metadata: MetaValue
    ) -> str | None:
        """A real operation is starting now. Returns the step id (None if emission failed)."""
        try:
            with self._lock:
                self._steps += 1
                step_id = f"s{self._steps}"
                self._open[step_id] = (self._clock(), kind, label)
            meta = {k: v for k, v in metadata.items() if v is not None}
            self._emit(
                RunEventType.STEP_STARTED,
                step_id=step_id,
                kind=kind,
                label=label,
                status=StepStatus.RUNNING,
                detail=detail,
                metadata=meta,
            )
            return step_id
        except Exception:  # noqa: BLE001 - never let reporting break the run
            self._warn()
            return None

    def finish(
        self,
        step_id: str | None,
        step: TraceStep,
        *,
        duration_ms: float | None = None,
        event_type: RunEventType | None = None,
    ) -> None:
        """The operation ended as ``step`` says. ``duration_ms`` defaults to the time since
        ``start`` (measured here); a step without a start is reported as already finished."""
        try:
            with self._lock:
                opened = self._open.pop(step_id, None) if step_id else None
                if step_id is None:
                    self._steps += 1
                    step_id = f"s{self._steps}"
            if duration_ms is None and opened is not None:
                duration_ms = round((self._clock() - opened[0]) * 1000, 1)
            status = _TRACE_TO_STEP[step.status]
            kind = event_type or (
                RunEventType.STEP_FAILED
                if status is StepStatus.FAILED
                else RunEventType.STEP_COMPLETED
            )
            self._emit(
                kind,
                step_id=step_id,
                kind=step.kind,
                label=step.label,
                status=status,
                detail=step.detail,
                metadata=dict(step.metadata),
                duration_ms=duration_ms,
            )
        except Exception:  # noqa: BLE001
            self._warn()

    def fail(
        self,
        step_id: str | None,
        kind: TraceEventKind,
        label: str,
        detail: str,
        **metadata: MetaValue,
    ) -> None:
        self.finish(step_id, TraceStep.of(kind, label, TraceStatus.FAILED, detail, **metadata))

    def fail_open_steps(self, detail: str) -> None:
        """The run stopped while these steps were running: report each one as failed (they
        really started and did not finish), never as completed."""
        with self._lock:
            open_steps = list(self._open.items())
        for step_id, (_, kind, label) in open_steps:
            self.fail(step_id, kind, label, detail)

    def skip_unused(self, capabilities: list[Capability], detail: str) -> None:
        """Capabilities of this run's graph that never ran: reported as skipped only now,
        once the run has ended (or paused on the fixed approval path)."""
        for cap in capabilities:
            if cap.kind not in self.kinds_seen:
                self.skip(cap.kind, cap.label, detail)

    def approval_required(self, step: TraceStep, next_steps: list[Capability]) -> None:
        try:
            with self._lock:
                self._steps += 1
                step_id = f"s{self._steps}"
            self._emit(
                RunEventType.APPROVAL_REQUIRED,
                step_id=step_id,
                kind=step.kind,
                label=step.label,
                status=StepStatus.WAITING,
                detail=step.detail,
                metadata=dict(step.metadata),
                next_steps=next_steps,
            )
        except Exception:  # noqa: BLE001
            self._warn()

    def skip(self, kind: TraceEventKind, label: str, detail: str) -> None:
        try:
            with self._lock:
                self._steps += 1
                step_id = f"s{self._steps}"
            self._emit(
                RunEventType.STEP_SKIPPED,
                step_id=step_id,
                kind=kind,
                label=label,
                status=StepStatus.SKIPPED,
                detail=detail,
            )
        except Exception:  # noqa: BLE001
            self._warn()

    # --- internals -----------------------------------------------------------------------------
    def _emit(self, event_type: RunEventType, **fields: Any) -> None:
        try:
            with self._lock:
                self._sequence += 1
                event = RunEvent(
                    type=event_type,
                    run_id=self.run_id,
                    sequence=self._sequence,
                    elapsed_ms=round((self._clock() - self._started) * 1000, 1),
                    **fields,
                )
                if event.kind is not None:
                    self.kinds_seen.add(event.kind)
                # Delivered under the lock: events reach the sink in sequence order.
                self._sink(event)
        except Exception:  # noqa: BLE001 - a broken sink/stream never affects the run
            self._warn()

    def _warn(self) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("run event emission failed", extra={"run_id": self.run_id[:12]})


class _NullEmitter(RunEmitter):
    """Used when nobody listens: no events, no cost beyond a method call."""

    def __init__(self) -> None:
        super().__init__(lambda _e: None, run_id="0" * 32)

    def start(self, *_a: Any, **_k: Any) -> str | None:  # type: ignore[override]
        return None

    def finish(self, *_a: Any, **_k: Any) -> None:  # type: ignore[override]
        return None

    def approval_required(self, *_a: Any, **_k: Any) -> None:  # type: ignore[override]
        return None

    def skip(self, *_a: Any, **_k: Any) -> None:  # type: ignore[override]
        return None

    def fail_open_steps(self, *_a: Any, **_k: Any) -> None:  # type: ignore[override]
        return None

    def skip_unused(self, *_a: Any, **_k: Any) -> None:  # type: ignore[override]
        return None

    def _emit(self, event_type: RunEventType, **fields: Any) -> None:
        return None


NULL_EMITTER: RunEmitter = _NullEmitter()
_CURRENT: ContextVar[RunEmitter] = ContextVar("commerceops_run_emitter", default=NULL_EMITTER)


def current_emitter() -> RunEmitter:
    return _CURRENT.get()


@contextmanager
def emitter_scope(emitter: RunEmitter | None) -> Iterator[RunEmitter]:
    token = _CURRENT.set(emitter or NULL_EMITTER)
    try:
        yield _CURRENT.get()
    finally:
        _CURRENT.reset(token)
