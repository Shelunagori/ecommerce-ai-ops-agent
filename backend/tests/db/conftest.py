"""Real-PostgreSQL fixtures.

Requires TEST_DATABASE_URL pointing at a DEDICATED database whose name ends in
``_test`` (the suite runs ``alembic downgrade base``). Skipped when unset.
"""

import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, make_url
from sqlalchemy.orm import Session, sessionmaker

import app.db.session as db_session_module
from app.api.deps import get_clock
from app.core.tenant import TenantContext
from app.db.url import normalize_database_url
from app.main import create_app
from scripts.seed_demo import BLUEPEAK, NORTHSTAR, seed, tenant_id_for
from tests.conftest import make_settings

BACKEND_DIR = Path(__file__).resolve().parents[2]

# Fixed "now" so derived states (overdue) are deterministic in tests.
FIXED_NOW = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)


def fixed_clock() -> datetime:
    return FIXED_NOW


def _test_database_url() -> str:
    raw = os.getenv("TEST_DATABASE_URL")
    if not raw:
        pytest.skip("TEST_DATABASE_URL not set (real PostgreSQL tests)")
    url = normalize_database_url(raw)
    database = make_url(url).database or ""
    if not database.endswith("_test"):
        pytest.exit(
            "Refusing to run: TEST_DATABASE_URL must name a dedicated database ending in '_test'",
            returncode=2,
        )
    return url


def alembic_config(url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    # ConfigParser interpolation: escape '%' in passwords.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def reset_and_seed(engine: Engine, cfg: Config) -> None:
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    with Session(engine) as session, session.begin():
        seed(session)


@pytest.fixture(scope="session")
def database_url() -> str:
    return _test_database_url()


@pytest.fixture(scope="session")
def db_engine(database_url: str) -> Iterator[Engine]:
    engine = create_engine(database_url, future=True)
    reset_and_seed(engine, alembic_config(database_url))
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    """Session inside an outer transaction that is always rolled back."""
    connection = db_engine.connect()
    outer = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        outer.rollback()
        connection.close()


@pytest.fixture(scope="session")
def tenant_a() -> TenantContext:
    return TenantContext(tenant_id_for(NORTHSTAR.slug))


@pytest.fixture(scope="session")
def tenant_b() -> TenantContext:
    return TenantContext(tenant_id_for(BLUEPEAK.slug))


@pytest.fixture
def api_app(db_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """App wired to the test DB through the REAL read-only session dependency."""
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    app = create_app(make_settings())
    app.dependency_overrides[get_clock] = lambda: fixed_clock
    return app


@pytest.fixture
def api(api_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(api_app) as client:
        yield client


@pytest.fixture
def headers_for() -> Callable[[TenantContext], dict[str, str]]:
    return lambda tenant: {"X-Tenant-ID": str(tenant.tenant_id)}


# --- committed writes (Step 10 actions) --------------------------------------------------------
_ACTION_TABLES = (
    "audit_events",
    "agent_runs",
    "store_credit_transactions",
    "action_requests",
    "tenant_memberships",
    "public_demo_usage",
)
_EXTRA_CLEANUP: list[str] = []  # later phases register their tenant-scoped audit tables here


@pytest.fixture
def committed(db_engine: Engine, monkeypatch: pytest.MonkeyPatch):
    """A REAL committing unit of work on the test DB (needed for locking/concurrency),
    wired into ``app.db.session``. After the test every action row is deleted and every
    order status is restored, so later tests see the pristine seed."""
    from contextlib import contextmanager

    from sqlalchemy import text

    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    with db_engine.connect() as conn:
        statuses = conn.execute(text("SELECT id, status FROM orders")).all()

    @contextmanager
    def scope():
        with factory() as session, session.begin():
            yield session

    try:
        yield scope
    finally:
        with db_engine.begin() as conn:
            for table in (*_EXTRA_CLEANUP, *_ACTION_TABLES):
                conn.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed table names
            for order_id, status in statuses:
                conn.execute(
                    text("UPDATE orders SET status = :s WHERE id = :i"),
                    {"s": status, "i": order_id},
                )
