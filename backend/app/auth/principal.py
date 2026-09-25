"""Principal resolution: the ONLY place that decides who is calling and for which tenant.

* ``demo`` mode (local development only; refused when APP_ENV=production): the tenant comes
  from the ``X-Tenant-ID`` header, as in the original demo API. NOT authentication.
* ``supabase`` mode: ``Authorization: Bearer <access token>`` is verified (``JwtVerifier``);
  the user's tenants come from ``tenant_memberships`` (server side). ``X-Tenant-ID`` is only a
  SELECTOR among the user's memberships; a tenant without membership is refused exactly like
  a non-existent one. The trusted ``AgentContext`` is built from the result, never from the
  request body or the model.
* public demo (``supabase`` mode + ``PUBLIC_DEMO_ENABLED``): a token whose VERIFIED claims
  carry ``is_anonymous: true`` (Supabase Anonymous Sign-Ins) is given exactly ONE membership,
  the configured demo tenant, as ``member`` and ``public_demo=True`` (read-only: no approvals,
  no action capability). No ``tenant_memberships`` rows are read or written for it. Nothing
  from the request (headers, body) can make a principal anonymous.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.errors import (
    AuthenticationRequiredError,
    AuthNotConfiguredError,
    PublicDemoDisabledError,
    PublicDemoUnavailableError,
    TenantForbiddenError,
    TenantSelectionRequiredError,
)
from app.auth.jwt import JwtVerifier
from app.core.errors import TenantContextInvalidError, TenantContextMissingError
from app.core.tenant import TenantContext
from app.models import Tenant, TenantMembership
from app.services.tenants import resolve_tenant_context

Role = Literal["member", "approver"]


@dataclass(frozen=True)
class Membership:
    tenant_id: uuid.UUID
    slug: str
    name: str
    role: Role


@dataclass(frozen=True)
class Principal:
    subject: str
    mode: Literal["demo", "supabase"]
    tenant: TenantContext
    role: Role
    memberships: tuple[Membership, ...]
    public_demo: bool = False  # verified anonymous visitor: read-only, one demo tenant

    @property
    def can_approve(self) -> bool:
        return self.role == "approver" and not self.public_demo

    @property
    def can_use_actions(self) -> bool:
        """Approval-gated write capabilities (proposals, action resources)."""
        return not self.public_demo


@dataclass(frozen=True)
class Identity:
    subject: str
    memberships: tuple[Membership, ...]
    public_demo: bool = False


def _parse_tenant(raw: str | None) -> uuid.UUID | None:
    if raw is None or not raw.strip():
        return None
    try:
        return uuid.UUID(raw.strip())
    except ValueError:
        raise TenantContextInvalidError() from None


def _bearer(authorization: str | None) -> str:
    if not authorization:
        raise AuthenticationRequiredError()
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationRequiredError()
    return token.strip()


def memberships_for(session: Session, subject: str) -> tuple[Membership, ...]:
    rows = session.execute(
        select(Tenant.id, Tenant.slug, Tenant.name, TenantMembership.role)
        .join(TenantMembership, TenantMembership.tenant_id == Tenant.id)
        .where(TenantMembership.user_subject == subject)
        .order_by(Tenant.slug)
    ).all()
    return tuple(Membership(r.id, r.slug, r.name, r.role) for r in rows)


def is_verified_anonymous(claims: dict) -> bool:
    """Only a boolean ``true`` in the VERIFIED claims counts (never "true", 1, a header...)."""
    return claims.get("is_anonymous") is True


def public_demo_membership(session: Session, settings: object) -> Membership:
    if not getattr(settings, "public_demo_enabled", False):
        raise PublicDemoDisabledError()
    slug = getattr(settings, "public_demo_tenant_slug", "")
    row = session.execute(
        select(Tenant.id, Tenant.slug, Tenant.name).where(Tenant.slug == slug)
    ).one_or_none()
    if row is None:  # misconfiguration: fail closed, never fall back to another tenant
        raise PublicDemoUnavailableError()
    return Membership(row.id, row.slug, row.name, "member")


def _identity(session: Session, claims: dict, settings: object) -> Identity:
    subject = claims["sub"]
    if is_verified_anonymous(claims):
        return Identity(subject, (public_demo_membership(session, settings),), public_demo=True)
    return Identity(subject, memberships_for(session, subject))


def resolve_principal(
    session: Session,
    settings: object,
    *,
    authorization: str | None,
    x_tenant_id: str | None,
    verifier: JwtVerifier | None = None,
) -> Principal:
    mode = getattr(settings, "auth_mode", "demo")
    if mode == "demo":
        if getattr(settings, "app_env", "development") == "production":
            raise AuthNotConfiguredError()  # demo header is never trusted in production
        tenant_id = _parse_tenant(x_tenant_id)
        if tenant_id is None:
            raise TenantContextMissingError()
        tenant = resolve_tenant_context(session, tenant_id)
        demo = tuple(
            Membership(r.id, r.slug, r.name, "approver")
            for r in session.execute(
                select(Tenant.id, Tenant.slug, Tenant.name).order_by(Tenant.slug)
            )
        )
        return Principal(settings.demo_user_subject, "demo", tenant, "approver", demo)  # type: ignore[attr-defined]

    claims = (verifier or JwtVerifier.from_settings(settings)).verify(_bearer(authorization))
    identity = _identity(session, claims, settings)
    subject, memberships = identity.subject, identity.memberships
    wanted = _parse_tenant(x_tenant_id)
    if wanted is None:
        if len(memberships) == 1:
            chosen = memberships[0]
        elif not memberships:
            raise TenantForbiddenError()
        else:
            raise TenantSelectionRequiredError()
    else:
        chosen = next((m for m in memberships if m.tenant_id == wanted), None)
        if chosen is None:
            raise TenantForbiddenError()
    return Principal(
        subject,
        "supabase",
        TenantContext(chosen.tenant_id),
        chosen.role,
        memberships,
        public_demo=identity.public_demo,
    )


def identity_only(
    session: Session,
    settings: object,
    *,
    authorization: str | None,
    verifier: JwtVerifier | None = None,
) -> Identity:
    """Who is calling and which tenants they may use (for the tenant selector)."""
    if getattr(settings, "auth_mode", "demo") == "demo":
        if getattr(settings, "app_env", "development") == "production":
            raise AuthNotConfiguredError()
        demo = tuple(
            Membership(r.id, r.slug, r.name, "approver")
            for r in session.execute(
                select(Tenant.id, Tenant.slug, Tenant.name).order_by(Tenant.slug)
            )
        )
        return Identity(settings.demo_user_subject, demo)  # type: ignore[attr-defined]
    claims = (verifier or JwtVerifier.from_settings(settings)).verify(_bearer(authorization))
    return _identity(session, claims, settings)
