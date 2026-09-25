"""Server-sent run events over a POST response (``text/event-stream``).

Why POST + fetch streaming (not the browser ``EventSource`` API): a run needs a JSON body
(the message), an ``Authorization: Bearer`` header and the ``X-Tenant-ID`` selector;
``EventSource`` can only issue a GET without custom headers. The browser reads this response
with ``fetch()`` + ``ReadableStream`` instead. The wire format is standard SSE:

    event: step_started
    id: 3
    data: {"type":"step_started","run_id":"…","sequence":3,…}

Execution model - the SAME synchronous ``assistant.run``/``resume`` as the JSON endpoints:
it runs in a worker thread with a ``RunEmitter`` whose sink hands events to the event loop
(``call_soon_threadsafe``); the response body drains that queue in order and sends a comment
heartbeat every 15 s so idle proxies keep the connection open.

Disconnect safety: the run is NOT tied to the connection. If the browser goes away, the
worker thread still finishes the run (a thread is never killed half-way), every write keeps
its own transaction/idempotency guarantees, and the outcome is in PostgreSQL (checkpointed
thread, action request + audit events) where history/approval endpoints recover it. Events
for a vanished client are dropped. Nothing is retried here, so a request runs at most once.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.assistant import AssistantError
from app.agent.events import RunEmitter, RunEvent
from app.core.errors import AppError

logger = logging.getLogger("app.api.stream")

HEARTBEAT_SECONDS = 15.0
SSE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Accel-Buffering": "no",  # proxies (nginx-style) must not buffer the stream
}


def encode_event(event: RunEvent) -> bytes:
    data = event.model_dump_json(exclude_none=True)
    return f"event: {event.type}\nid: {event.sequence}\ndata: {data}\n\n".encode()


def _failure(exc: BaseException) -> tuple[str, str, int]:
    """Safe code/message/status for a failed run (same envelope as the JSON API)."""
    if isinstance(exc, AssistantError):
        from app.api.routes.agent import assistant_status  # noqa: PLC0415 - avoid a cycle

        return exc.code, exc.message, assistant_status(exc)
    if isinstance(exc, AppError):
        return exc.code, exc.message, exc.status_code
    logger.error("streamed run failed", exc_info=exc)  # traceback in server logs only
    return AppError.code, AppError.message, 500


def stream_run(
    work: Callable[[RunEmitter], BaseModel],
    *,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> StreamingResponse:
    """Run ``work(emitter)`` in a worker thread and stream its events.

    ``work`` returns the normal API response model; it is sent as ``run_completed.response``.
    Any failure becomes ``run_failed`` with the API's safe error code/message/status."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()

    def deliver(item: RunEvent | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)  # raises only if the loop is gone

    emitter = RunEmitter(deliver)

    def job() -> None:
        try:
            response = work(emitter)
        except BaseException as exc:  # noqa: BLE001 - reported as a terminal event
            code, message, status = _failure(exc)
            emitter.run_failed(code, message, status=status)
        else:
            emitter.run_completed(response.model_dump(mode="json"))
        finally:
            try:
                deliver(None)
            except RuntimeError:  # event loop already closed (shutdown): nobody is listening
                pass

    ctx = contextvars.copy_context()  # request id for logs/audit correlation
    future = loop.run_in_executor(None, ctx.run, job)
    future.add_done_callback(_consume)

    async def body() -> AsyncIterator[bytes]:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), heartbeat_seconds)
            except TimeoutError:
                yield b": keep-alive\n\n"
                continue
            if item is None:
                return
            yield encode_event(item)

    return StreamingResponse(body(), media_type="text/event-stream", headers=SSE_HEADERS)


def _consume(future: Any) -> None:
    if not future.cancelled() and future.exception() is not None:  # pragma: no cover
        logger.error("stream worker crashed", exc_info=future.exception())
