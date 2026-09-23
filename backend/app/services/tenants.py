"""Tenant resolution: the single place that turns a tenant id into a TenantContext."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import TenantNotFoundError
from app.core.tenant import TenantContext
from app.models import Tenant


def resolve_tenant_context(session: Session, tenant_id: uuid.UUID) -> TenantContext:
    """Return a TenantContext for an existing tenant or raise TenantNotFoundError.

    Callers decide *how* the tenant id was established (demo header today,
    authenticated identity later); this function only checks that it exists.
    """
    exists = session.scalar(select(Tenant.id).where(Tenant.id == tenant_id))
    if exists is None:
        raise TenantNotFoundError()
    return TenantContext(tenant_id=tenant_id)
