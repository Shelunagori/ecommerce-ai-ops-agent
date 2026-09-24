import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.request_context import (
    loggable_tenant_id,
    request_id_var,
    resolve_request_id,
    tenant_id_var,
)

logger = logging.getLogger("app.request")

# Phase 12: a JSON API is never framed, sniffed or referred; tenant data is never cached.
SECURITY_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
)
NO_STORE = (b"cache-control", b"no-store")


class RequestContextMiddleware:
    """Assign a request id, expose it as X-Request-ID, and log one line per request.

    The tenant id logged here is the *claimed* one from the demo header (only if it is
    a well-formed UUID). Query strings and bodies are not logged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        request_id = resolve_request_id(headers.get("x-request-id"))
        rid_token = request_id_var.set(request_id)
        tid_token = tenant_id_var.set(loggable_tenant_id(headers.get("x-tenant-id")))
        status_code = 500
        response_started = False
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_started = True
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", request_id.encode()))
                present = {k.lower() for k, _ in message["headers"]}
                extra = [*SECURITY_HEADERS]
                if scope["path"].startswith("/api/"):
                    extra.append(NO_STORE)
                message["headers"].extend(h for h in extra if h[0] not in present)
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception as exc:
            # Unhandled errors are answered HERE (inside the request context) so the generic
            # 500 still carries the request id and security headers. Traceback: logs only.
            if response_started:
                raise
            from app.api.errors import unhandled_error_response  # noqa: PLC0415

            await unhandled_error_response(exc)(scope, receive, send_with_request_id)
        finally:
            logger.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            request_id_var.reset(rid_token)
            tenant_id_var.reset(tid_token)
