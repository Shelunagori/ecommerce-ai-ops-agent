import logging
from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from app.db.readiness import readiness_checks
from app.db.session import DatabaseNotConfiguredError, ping_database
from app.schemas.health import DatabaseHealthResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

SERVICE_NAME = "commerceops-api"

DbCheck = Callable[[], None]


def get_db_check() -> DbCheck:
    """Dependency returning the DB check; overridden in tests."""
    return ping_database


def get_readiness_check() -> Callable[..., dict[str, str]]:
    """Dependency returning the readiness probe; overridden in tests."""
    return readiness_checks


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


@router.get("/ready", responses={503: {"description": "not ready"}})
def health_ready(
    request: Request,
    check: Annotated[Callable[..., dict[str, str]], Depends(get_readiness_check)],
) -> JSONResponse:
    """Readiness for load balancers / Railway: config, database, migrations, checkpoints.
    Codes only; never hostnames, versions or error text."""
    checks = check(request.app.state.settings)
    ready = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )
