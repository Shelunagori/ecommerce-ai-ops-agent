"""Alembic round trip on the dedicated *_test database: base -> head -> base -> head."""

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.models import Base
from scripts.seed_demo import seed
from tests.db.conftest import alembic_config, reset_and_seed

DOMAIN_TABLES = {
    "tenants",
    "customers",
    "products",
    "orders",
    "order_items",
    "invoices",
    "shipments",
    "knowledge_documents",  # Step 7
    "knowledge_chunks",  # Step 7
}


def _tables(engine) -> set[str]:
    return set(inspect(engine).get_table_names()) - {"alembic_version"}


def _current(engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def test_migration_round_trip_and_models_match(db_engine, database_url):
    cfg = alembic_config(database_url)
    head = ScriptDirectory.from_config(cfg).get_current_head()
    try:
        assert _current(db_engine) == head

        command.downgrade(cfg, "base")
        assert _tables(db_engine) == set()
        assert _current(db_engine) is None

        command.upgrade(cfg, "head")
        assert _tables(db_engine) == DOMAIN_TABLES
        assert _current(db_engine) == head

        # The migration and the ORM models describe the same schema.
        with db_engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        assert diff == []

        # No extension (e.g. vector) is enabled by this step.
        with db_engine.connect() as conn:
            exts = set(conn.exec_driver_sql("SELECT extname FROM pg_extension").scalars())
        assert exts == {"plpgsql"}
    finally:
        # Leave the shared test database migrated and seeded for other tests.
        reset_and_seed(db_engine, cfg)


ECOMMERCE_TABLES = sorted(DOMAIN_TABLES - {"knowledge_documents", "knowledge_chunks"})


def _ecommerce_snapshot(engine) -> dict[str, list[tuple]]:
    with engine.connect() as conn:
        return {
            t: [tuple(r) for r in conn.exec_driver_sql(f"SELECT * FROM {t} ORDER BY id")]  # noqa: S608
            for t in ECOMMERCE_TABLES
        }


def test_knowledge_migration_leaves_ecommerce_rows_untouched(db_engine, database_url):
    cfg = alembic_config(database_url)
    try:
        command.downgrade(cfg, "0001")
        assert "knowledge_documents" not in _tables(db_engine)
        with Session(db_engine) as session, session.begin():
            seed(session)
        before = _ecommerce_snapshot(db_engine)
        assert all(before[t] for t in ECOMMERCE_TABLES)

        command.upgrade(cfg, "0002")
        assert _ecommerce_snapshot(db_engine) == before
        command.downgrade(cfg, "0001")
        assert _ecommerce_snapshot(db_engine) == before
        command.upgrade(cfg, "head")
        assert _ecommerce_snapshot(db_engine) == before
        with db_engine.connect() as conn:
            exts = set(conn.exec_driver_sql("SELECT extname FROM pg_extension").scalars())
            cols = set(
                conn.exec_driver_sql(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'knowledge_chunks'"
                ).scalars()
            )
        assert exts == {"plpgsql"}  # no pgvector yet
        assert not {"embedding", "vector"} & cols
    finally:
        reset_and_seed(db_engine, cfg)
