"""Readiness probe (Phase 11): is this process able to serve agent traffic?

Each check reports a stable code only - no hostnames, versions or driver messages:

* ``config``      - ``configuration_problems`` is empty (always ok outside production)
* ``database``    - a read-only ``SELECT 1`` succeeds
* ``migrations``  - ``alembic_version`` equals the head revision shipped with this code
* ``checkpoints`` - the LangGraph PostgreSQL checkpoint tables exist (``setup_checkpoints``)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from sqlalchemy import text

from app.core.config import Settings
from app.core.production import configuration_problems
from app.db.session import DatabaseNotConfiguredError, read_only_session

logger = logging.getLogger("app.health")

ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
CHECKPOINT_TABLES = (
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
)


@lru_cache
def migration_head() -> str:
    from alembic.config import Config  # noqa: PLC0415
    from alembic.script import ScriptDirectory  # noqa: PLC0415

    cfg = Config()
    cfg.set_main_option("script_location", str(ALEMBIC_DIR))
    heads = ScriptDirectory.from_config(cfg).get_heads()
    if len(heads) != 1:  # pragma: no cover - guarded by the migration tests
        raise RuntimeError("expected exactly one alembic head")
    return heads[0]


def _query(sql: str, **params: object) -> object:
    with read_only_session() as session:
        return session.execute(text(sql), params).scalar()


def readiness_checks(settings: Settings) -> dict[str, str]:
    checks = {"config": "failed" if configuration_problems(settings) else "ok"}
    try:
        _query("SELECT 1")
        checks["database"] = "ok"
    except DatabaseNotConfiguredError:
        checks["database"] = "not_configured"
    except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
        logger.warning(
            "readiness check failed", extra={"check": "database", "error_type": type(exc).__name__}
        )
        checks["database"] = "failed"
    if checks["database"] != "ok":
        checks["migrations"] = checks["checkpoints"] = "unknown"
        return checks
    try:
        current = _query("SELECT version_num FROM alembic_version")
        checks["migrations"] = "ok" if current == migration_head() else "not_at_head"
    except Exception as exc:  # noqa: BLE001 - missing table or unreadable
        logger.warning(
            "readiness check failed",
            extra={"check": "migrations", "error_type": type(exc).__name__},
        )
        checks["migrations"] = "not_at_head"
    try:
        present = _query(
            "SELECT count(*) FROM unnest(CAST(:names AS text[])) AS n "
            "WHERE to_regclass('public.' || n) IS NOT NULL",
            names=list(CHECKPOINT_TABLES),
        )
        checks["checkpoints"] = "ok" if present == len(CHECKPOINT_TABLES) else "missing"
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "readiness check failed",
            extra={"check": "checkpoints", "error_type": type(exc).__name__},
        )
        checks["checkpoints"] = "failed"
    return checks
