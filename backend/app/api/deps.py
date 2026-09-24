"""Request-scoped dependencies: settings, identity/tenant boundary, sessions, queries."""

from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from app.auth.jwt import JwtVerifier
from app.auth.principal import Principal, resolve_principal
from app.core.config import Settings, get_settings
from app.core.tenant import TenantContext
from app.db.session import get_read_session
from app.services import CommerceQueries
from app.services.base import Clock, utc_now

ReadSession = Annotated[Session, Depends(get_read_session)]

TenantHeader = Annotated[
    str | None,
    Header(
        alias="X-Tenant-ID",
        description=(
            "Tenant selector. demo mode: trusted as-is (LOCAL DEMO ONLY). supabase mode: must "
            "be one of the authenticated user's tenant memberships."
        ),
    ),
]
AuthorizationHeader = Annotated[str | None, Header(alias="Authorization")]


def get_app_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


def get_jwt_verifier(request: Request) -> JwtVerifier | None:
    settings = get_app_settings(request)
    if settings.auth_mode != "supabase":
        return None
    verifier = getattr(request.app.state, "jwt_verifier", None)
    if verifier is None:
        verifier = JwtVerifier.from_settings(settings)  # no network until first verify
        request.app.state.jwt_verifier = verifier
    return verifier


def get_principal(
    request: Request,
    session: ReadSession,
    x_tenant_id: TenantHeader = None,
    authorization: AuthorizationHeader = None,
) -> Principal:
    """The ONLY place a request's identity and tenant are established (app.auth.principal)."""
    return resolve_principal(
        session,
        get_app_settings(request),
        authorization=authorization,
        x_tenant_id=x_tenant_id,
        verifier=get_jwt_verifier(request),
    )


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


def get_tenant_context(principal: CurrentPrincipal) -> TenantContext:
    return principal.tenant


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
