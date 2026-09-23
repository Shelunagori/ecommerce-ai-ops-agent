"""Execution plumbing shared by every commerce tool.

Each invocation: validate the injected AgentContext -> open its OWN read-only session ->
build the tenant-scoped query facade -> run one operation -> roll back and close. No
session is ever shared, so concurrent tool calls are safe.
"""

import logging
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from langchain.tools import BaseTool, ToolRuntime
from pydantic import ValidationError
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session

from app.agent.context import AgentContext
from app.agent.tools import envelope
from app.agent.tools.envelope import ErrorCode
from app.core.errors import NotFoundError
from app.core.request_context import request_id_var, tenant_id_var
from app.db.session import DatabaseNotConfiguredError, read_only_session
from app.services import CommerceQueries
from app.services.base import Clock, utc_now

logger = logging.getLogger("app.agent.tools")

SessionScope = Callable[[], AbstractContextManager[Session]]

_UNAVAILABLE = (OperationalError, InterfaceError, PoolTimeoutError, DatabaseNotConfiguredError)


@dataclass(frozen=True)
class ToolDependencies:
    """Injected infrastructure. Defaults are production; tests swap them."""

    session_scope: SessionScope = field(default=read_only_session)
    clock: Clock = field(default=utc_now)


def execute(
    tool_name: str,
    runtime: ToolRuntime[AgentContext],
    deps: ToolDependencies,
    operation: Callable[[CommerceQueries], Any],
) -> envelope.Envelope:
    started = time.perf_counter()
    context = getattr(runtime, "context", None)
    if not isinstance(context, AgentContext):
        _log(tool_name, None, "internal_error", ErrorCode.INTERNAL_ERROR, started, "no_context")
        return envelope.failure(ErrorCode.INTERNAL_ERROR)

    rid = request_id_var.set(context.request_id) if context.request_id else None
    tid = tenant_id_var.set(str(context.tenant_id))
    try:
        try:
            with deps.session_scope() as session:
                data = operation(CommerceQueries(session, context.tenant, deps.clock))
            result, outcome, code = envelope.success(data), "ok", None
        except NotFoundError as exc:
            result, outcome, code = envelope.failure(exc.code, exc.message), "not_found", exc.code
        except _UNAVAILABLE as exc:
            code = ErrorCode.SERVICE_UNAVAILABLE
            result, outcome = envelope.failure(code), "unavailable"
            _log(tool_name, context, outcome, code, started, type(exc).__name__)
            return result
        except Exception as exc:  # noqa: BLE001 - nothing unexpected may reach the model
            code = ErrorCode.INTERNAL_ERROR
            result, outcome = envelope.failure(code), "internal_error"
            # Exception type only in structured fields; traceback for operators at DEBUG.
            logger.debug("tool internal error", exc_info=exc)
            _log(tool_name, context, outcome, code, started, type(exc).__name__)
            return result
        _log(tool_name, context, outcome, code, started)
        return result
    finally:
        tenant_id_var.reset(tid)
        if rid is not None:
            request_id_var.reset(rid)


def validation_error_handler(tool_name: str) -> Callable[[ValidationError], str]:
    """Turn argument validation failures into the stable envelope (no input echoed).

    LangChain's ``handle_validation_error`` callback must return ``str``, so this one
    boundary returns the standard envelope serialised to JSON.
    """

    def handle(error: ValidationError) -> str:
        problems = error.errors()
        if any(p.get("loc", ("",))[0] == "runtime" for p in problems):
            # Missing/forged runtime is a host bug or an injection attempt, not a model typo.
            _log(tool_name, None, "internal_error", ErrorCode.INTERNAL_ERROR, None, "bad_runtime")
            return envelope.failure_json(ErrorCode.INTERNAL_ERROR)
        fields = sorted(
            {f"{'.'.join(str(x) for x in p.get('loc', ()))} ({p.get('type')})" for p in problems}
        )
        _log(tool_name, None, "invalid_arguments", ErrorCode.INVALID_ARGUMENTS, None)
        return envelope.failure_json(
            ErrorCode.INVALID_ARGUMENTS, f"Invalid arguments: {', '.join(fields)}."
        )

    return handle


def finalize(tool: BaseTool, *, domain: str) -> BaseTool:
    """Common settings for every registered tool."""
    tool.handle_validation_error = validation_error_handler(tool.name)
    tool.metadata = {"read_only": True, "domain": domain}
    tool.tags = ["commerce", "read_only", domain]
    return tool


def _log(
    tool_name: str,
    context: AgentContext | None,
    outcome: str,
    error_code: str | None,
    started: float | None,
    error_type: str | None = None,
) -> None:
    fields: dict[str, Any] = {"tool": tool_name, "outcome": outcome}
    # Explicit on the record so correlation never depends on handler/filter configuration.
    if context is not None:
        fields["tenant_id"] = str(context.tenant_id)
        if context.request_id:
            fields["request_id"] = context.request_id
    if error_code:
        fields["error_code"] = error_code
    if error_type:
        fields["error_type"] = error_type
    if started is not None:
        fields["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
    level = logging.WARNING if outcome in ("unavailable", "internal_error") else logging.INFO
    logger.log(level, "tool call", extra=fields)
