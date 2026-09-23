"""Request-scoped dependencies for the read-only demo API."""

import uuid
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from app.core.errors import TenantContextInvalidError, TenantContextMissingError
from app.core.tenant import TenantContext
from app.db.session import get_read_session
from app.services import CommerceQueries
from app.services.base import Clock, utc_now
from app.services.tenants import resolve_tenant_context

ReadSession = Annotated[Session, Depends(get_read_session)]


def get_tenant_context(
    session: ReadSession,
    x_tenant_id: Annotated[
        str | None,
        Header(
            alias="X-Tenant-ID",
            description=(
                "Demo tenant selector. NOT authentication: any caller can send any value. "
                "Will be replaced by authenticated tenant context."
            ),
        ),
    ] = None,
) -> TenantContext:
    """TEMPORARY demo mechanism: the tenant comes from a client-supplied header.

    This is the only place that decides *how* a request's tenant is established.
    Replacing it with authenticated identity later does not affect the services.
    """
    if x_tenant_id is None or not x_tenant_id.strip():
        raise TenantContextMissingError()
    try:
        tenant_id = uuid.UUID(x_tenant_id.strip())
    except ValueError:
        raise TenantContextInvalidError() from None
    return resolve_tenant_context(session, tenant_id)


def get_clock() -> Clock:
    """Overridable time source (derived states like 'overdue' depend on it)."""
    return utc_now


def get_queries(
    session: ReadSession,
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
    clock: Annotated[Clock, Depends(get_clock)],
) -> CommerceQueries:
    return CommerceQueries(session, tenant, clock)


Queries = Annotated[CommerceQueries, Depends(get_queries)]
