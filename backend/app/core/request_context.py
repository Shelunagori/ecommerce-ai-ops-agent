"""Per-request identifiers for logging (request id, claimed tenant id)."""

import re
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def resolve_request_id(incoming: str | None) -> str:
    """Accept a well-formed inbound X-Request-ID, otherwise mint one."""
    if incoming and _SAFE_REQUEST_ID.fullmatch(incoming):
        return incoming
    return uuid.uuid4().hex


def loggable_tenant_id(raw: str | None) -> str | None:
    """Only well-formed UUIDs are logged; arbitrary header content never reaches logs."""
    if not raw:
        return None
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return None
