import logging
from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.db.session import DatabaseNotConfiguredError, ping_database
from app.schemas.health import DatabaseHealthResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

SERVICE_NAME = "commerceops-api"

DbCheck = Callable[[], None]


def get_db_check() -> DbCheck:
    """Dependency returning the DB check; overridden in tests."""
    return ping_database


@router.get("", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME)


@router.get(
    "/db",
    response_model=DatabaseHealthResponse,
    responses={503: {"model": DatabaseHealthResponse}},
)
def health_db(
    check: Annotated[DbCheck, Depends(get_db_check)],
) -> DatabaseHealthResponse | JSONResponse:
    try:
        check()
    except DatabaseNotConfiguredError:
        logger.warning("database health check failed", extra={"reason": "not_configured"})
        return _unavailable("not_configured")
    except Exception as exc:  # any failure means "unreachable"
        # Log the exception type only: driver messages can include host/user details.
        logger.warning(
            "database health check failed",
            extra={"reason": "unreachable", "error_type": type(exc).__name__},
        )
        return _unavailable("unreachable")
    return DatabaseHealthResponse(status="ok", database="reachable")


def _unavailable(database: Literal["unreachable", "not_configured"]) -> JSONResponse:
    body = DatabaseHealthResponse(status="error", database=database)
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump())
