import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_error_handlers
from app.api.health import router as health_router
from app.api.middleware import RequestContextMiddleware
from app.api.routes.commerce import router as commerce_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.debug)

    app = FastAPI(title="CommerceOps AI API", version="0.1.0", debug=settings.debug)

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

    logger.info(
        "application configured",
        extra={"app_env": settings.app_env, "cors_origins": origins},
    )
    return app


app = create_app()
