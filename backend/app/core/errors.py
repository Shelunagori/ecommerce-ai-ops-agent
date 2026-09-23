"""Safe, structured application errors.

Messages are written for API clients: they never contain SQL, connection details,
stack traces or data belonging to another tenant. A "not found" is reported
identically whether the reference does not exist at all or exists in another tenant.
"""


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        if message is not None:
            self.message = message
        super().__init__(self.message)


class NotFoundError(AppError):
    status_code = 404

    def __init__(self, resource: str, reference: str) -> None:
        self.resource = resource
        self.reference = reference
        self.code = f"{resource}_not_found"
        super().__init__(f"{resource.capitalize()} '{reference}' was not found.")


class TenantContextMissingError(AppError):
    status_code = 400
    code = "tenant_context_missing"
    message = "The X-Tenant-ID header is required."


class TenantContextInvalidError(AppError):
    status_code = 400
    code = "tenant_context_invalid"
    message = "The X-Tenant-ID header must be a UUID."


class TenantNotFoundError(AppError):
    status_code = 404
    code = "tenant_not_found"
    message = "The requested tenant was not found."
