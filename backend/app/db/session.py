"""SQLAlchemy engine/session infrastructure (plain PostgreSQL, provider-agnostic)."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.url import normalize_database_url


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when DATABASE_URL is not set."""


@lru_cache
def get_engine() -> Engine:
    """Create the engine lazily so the app can boot without a database."""
    url = get_settings().database_url
    if not url:
        raise DatabaseNotConfiguredError("DATABASE_URL is not set")
    return create_engine(
        normalize_database_url(url),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args={"connect_timeout": 5},
    )


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request (for later steps)."""
    with _session_factory()() as session:
        yield session


def ping_database() -> None:
    """Run a trivial query; raises on any connectivity problem."""
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
