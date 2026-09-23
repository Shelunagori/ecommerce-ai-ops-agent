"""SQLAlchemy engine/session infrastructure (plain PostgreSQL, provider-agnostic)."""

from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def read_only_session() -> Iterator[Session]:
    """Session whose transaction is READ ONLY at the database level and always rolled back.

    Use for every read path (API reads, future agent read tools). Any accidental write
    fails in PostgreSQL instead of silently persisting.
    """
    with _session_factory()() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        try:
            yield session
        finally:
            session.rollback()


@contextmanager
def unit_of_work() -> Iterator[Session]:
    """Read-write session: commits once on success, rolls back on any exception.

    The caller that opens the unit of work owns the transaction. Services and query
    classes never call ``commit()`` themselves, so operations compose safely.
    """
    with _session_factory()() as session, session.begin():
        yield session


def get_read_session() -> Iterator[Session]:
    """FastAPI dependency: one read-only session per request."""
    with read_only_session() as session:
        yield session


def ping_database() -> None:
    """Run a trivial query; raises on any connectivity problem."""
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
