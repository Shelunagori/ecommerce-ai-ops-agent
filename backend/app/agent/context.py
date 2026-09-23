"""Trusted runtime context for agent tool execution.

The tenant is an authorization/runtime concern: it is established by trusted code
(today the dev CLI; later the authenticated request) and handed to tools through
LangChain's ``ToolRuntime.context``. It is never a model-visible tool argument.
"""

import re
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.tenant import TenantContext
from app.services.tenants import resolve_tenant_context

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True, slots=True)
class AgentContext:
    tenant_id: uuid.UUID
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, uuid.UUID):
            raise TypeError("tenant_id must be a uuid.UUID")
        if self.request_id is not None and not _SAFE_REQUEST_ID.fullmatch(self.request_id):
            raise ValueError("request_id must be 1-64 chars of [A-Za-z0-9._-]")

    @property
    def tenant(self) -> TenantContext:
        return TenantContext(self.tenant_id)


def create_agent_context(
    session: Session, tenant_id: uuid.UUID, request_id: str | None = None
) -> AgentContext:
    """The trusted context-creation boundary: verifies the tenant exists (H1).

    Raises ``TenantNotFoundError`` for an unknown tenant. Tools themselves trust the
    resulting context and do not re-check it on every call.
    """
    resolve_tenant_context(session, tenant_id)
    return AgentContext(tenant_id=tenant_id, request_id=request_id)
