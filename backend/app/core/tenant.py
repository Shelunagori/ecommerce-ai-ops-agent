"""Tenant context: *which* tenant a unit of work runs as.

This is deliberately separate from *how* the tenant is established. Today the demo
API reads an ``X-Tenant-ID`` header (see ``app.api.deps``) - that is NOT
authentication. Later, authenticated identity can produce the same ``TenantContext``
without any change to the domain services, which only ever receive this object.
"""

import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TenantContext:
    tenant_id: uuid.UUID
