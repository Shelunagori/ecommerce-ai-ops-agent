"""Readiness probe on real PostgreSQL (Phase 11): migrations at head, checkpoint tables."""

import pytest
from sqlalchemy import text

from app.agent.graph.checkpoint import durable_checkpointer
from app.db import readiness


@pytest.fixture
def checkpoint_tables(database_url, db_engine):
    cp = durable_checkpointer(database_url, min_size=1, max_size=1)
    try:
        yield cp
    finally:
        cp.close()
        with db_engine.begin() as conn:  # leave the DB as the migration tests expect
            for table in readiness.CHECKPOINT_TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))  # noqa: S608


def test_ready_only_after_migrations_and_checkpoint_setup(api, checkpoint_tables):
    r = api.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["checks"] == {
        "config": "ok",
        "database": "ok",
        "migrations": "ok",
        "checkpoints": "missing",
    }
    checkpoint_tables.setup()
    r = api.get("/health/ready")
    assert (r.status_code, r.json()["status"]) == (200, "ready")


def test_schema_behind_the_code_is_not_ready(api, monkeypatch):
    monkeypatch.setattr(readiness, "migration_head", lambda: "9999_future")
    assert api.get("/health/ready").json()["checks"]["migrations"] == "not_at_head"


def test_head_matches_the_migration_files():
    assert readiness.migration_head().startswith("0006")


def _tables(db_engine):
    with db_engine.connect() as conn:
        return conn.execute(
            text("SELECT count(*) FROM pg_tables WHERE tablename LIKE 'checkpoint%'")
        ).scalar()


def test_predeploy_releases_a_ready_database(
    api, database_url, db_engine, checkpoint_tables, monkeypatch, capsys
):
    from app.core.config import get_settings
    from scripts import predeploy

    monkeypatch.chdir("/")  # no .env
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("DATABASE_URL", database_url)
    get_settings.cache_clear()
    try:
        assert predeploy.main([]) == 0
        assert predeploy.main([]) == 0  # idempotent
    finally:
        get_settings.cache_clear()
    out = capsys.readouterr()
    assert "pre-deploy complete" in out.out and database_url not in out.out + out.err
    assert api.get("/health/ready").json()["status"] == "ready"


def test_predeploy_aborts_before_touching_the_database(
    database_url, db_engine, checkpoint_tables, monkeypatch, capsys
):
    from app.core.config import get_settings
    from scripts import predeploy

    monkeypatch.chdir("/")
    monkeypatch.setenv("APP_ENV", "production")  # demo auth, local providers -> refused
    monkeypatch.setenv("DATABASE_URL", database_url)
    get_settings.cache_clear()
    try:
        assert predeploy.main([]) == 1
    finally:
        get_settings.cache_clear()
    assert "aborted at step: check_env" in capsys.readouterr().err
    assert _tables(db_engine) == 0
