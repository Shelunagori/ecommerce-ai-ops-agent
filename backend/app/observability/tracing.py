"""Optional LangSmith tracing. Off unless ``LANGSMITH_TRACING=true`` AND an API key is set;
the application never requires it. LangChain reads these process environment variables."""

import logging
import os

from app.core.config import Settings

logger = logging.getLogger("app.observability")


def configure_tracing(settings: Settings) -> bool:
    key = settings.langsmith_api_key.get_secret_value() if settings.langsmith_api_key else ""
    if settings.langsmith_tracing and key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info("langsmith tracing enabled", extra={"project": settings.langsmith_project})
        return True
    os.environ["LANGSMITH_TRACING"] = "false"  # explicit: no accidental tracing
    if settings.langsmith_tracing:
        logger.warning("langsmith tracing requested without an API key; left disabled")
    return False
