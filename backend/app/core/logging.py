"""Structured (JSON-lines) logging on top of the standard library."""

import json
import logging
import sys
from datetime import UTC, datetime

from app.core.request_context import request_id_var, tenant_id_var

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Anything passed via `extra=` becomes a top-level field.
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class RequestContextFilter(logging.Filter):
    """Attach request_id / tenant_id (when known) to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = request_id_var.get()
        tenant_id = tenant_id_var.get()
        if request_id is not None:
            record.request_id = request_id
        if tenant_id is not None:
            record.tenant_id = tenant_id
        return True


def configure_logging(debug: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestContextFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    # HTTP client libraries log full request URLs at INFO (a hosted provider's URL can carry
    # an account id); the app logs its own safe per-call line instead.
    for noisy in ("httpx", "httpcore", "openai", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
