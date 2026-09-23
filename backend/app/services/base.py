"""Shared plumbing for tenant-scoped query classes."""

from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.core.tenant import TenantContext

Clock = Callable[[], datetime]

MAX_LIMIT = 100
DEFAULT_LIMIT = 50


def utc_now() -> datetime:
    return datetime.now(UTC)


def clamp_limit(limit: int) -> int:
    return max(1, min(limit, MAX_LIMIT))


def clamp_offset(offset: int) -> int:
    return max(0, offset)


def like_contains(term: str) -> str:
    """ILIKE pattern for a substring match with user wildcards escaped."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class TenantScopedQueries:
    """Base for query classes. The tenant is fixed at construction time, so every public
    method is tenant-scoped by construction; there is no way to call one without it.

    Query classes never commit or flush: transaction ownership stays with the caller
    (``read_only_session`` / ``unit_of_work`` in ``app.db.session``).
    """

    def __init__(self, session: Session, tenant: TenantContext, clock: Clock = utc_now) -> None:
        if not isinstance(tenant, TenantContext):
            raise TypeError("tenant must be a TenantContext")
        self._session = session
        self._tenant = tenant
        self._tenant_id = tenant.tenant_id
        self._clock = clock
