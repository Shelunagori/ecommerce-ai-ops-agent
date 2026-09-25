"""Stable, safe authentication / authorisation errors."""

from app.core.errors import AppError


class AuthenticationRequiredError(AppError):
    status_code = 401
    code = "auth_required"
    message = "Authentication is required."


class InvalidTokenError(AppError):
    status_code = 401
    code = "auth_invalid"
    message = "The access token is invalid or expired."


class TenantForbiddenError(AppError):
    """Identical for 'tenant does not exist' and 'no membership' (no tenant enumeration)."""

    status_code = 403
    code = "tenant_forbidden"
    message = "You do not have access to this tenant."


class TenantSelectionRequiredError(AppError):
    status_code = 400
    code = "tenant_selection_required"
    message = "Select a tenant with the X-Tenant-ID header."


class AuthNotConfiguredError(AppError):
    status_code = 503
    code = "auth_not_configured"
    message = "Authentication is not configured on this server."


class PublicDemoDisabledError(AppError):
    status_code = 403
    code = "public_demo_disabled"
    message = "The public demo is not enabled on this server."


class PublicDemoUnavailableError(AppError):
    status_code = 503
    code = "public_demo_unavailable"
    message = "The public demo is temporarily unavailable."


class PublicDemoReadOnlyError(AppError):
    status_code = 403
    code = "public_demo_read_only"
    message = "The public demo is read-only. Sign in with a reviewer account to use actions."


class PublicDemoLimitReachedError(AppError):
    status_code = 429
    code = "public_demo_limit_reached"
    message = "Public demo limit reached. Please start a reviewer session or try again later."
