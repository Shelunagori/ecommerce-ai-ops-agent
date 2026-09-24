import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_error_handlers
from app.api.health import router as health_router
from app.api.middleware import RequestContextMiddleware
from app.api.routes.agent import router as agent_router
from app.api.routes.commerce import router as commerce_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.production import require_valid_production_configuration
from app.observability.tracing import configure_tracing

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Fail fast: a production process with an unsafe configuration never starts serving.
    require_valid_production_configuration(app.state.settings)
    yield
    runtime = getattr(app.state, "agent_runtime", None)
    if runtime is not None:  # close the durable checkpoint pool
        runtime.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.debug)
    configure_tracing(settings)

    # Phase 12: no public schema / interactive docs in production.
    public_docs = settings.app_env != "production"
    app = FastAPI(
        title="CommerceOps AI API",
        version="0.2.0",
        debug=settings.debug,
        lifespan=_lifespan,
        docs_url="/docs" if public_docs else None,
        redoc_url="/redoc" if public_docs else None,
        openapi_url="/openapi.json" if public_docs else None,
    )
    app.state.settings = settings

    origins = settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Browsers reject "*" with credentials; only allow credentials for explicit origins.
        allow_credentials="*" not in origins,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    # Outermost: request id/logging wraps everything, including CORS and error handling.
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)

    app.include_router(health_router)
    app.include_router(commerce_router)
    app.include_router(agent_router)

    logger.info(
        "application configured",
        extra={"app_env": settings.app_env, "cors_origins": origins},
    )
    return app


app = create_app()
