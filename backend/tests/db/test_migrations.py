"""Alembic round trip on the dedicated *_test database: base -> head -> base -> head."""

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.knowledge.chunking import ChunkingConfig
from app.knowledge.embeddings.materialize import materialize_embeddings
from app.knowledge.ingest import ingest_policies
from app.knowledge.sources import DEFAULT_POLICY_DIR
from app.models import Base
from scripts.seed_demo import seed
from tests.db.conftest import alembic_config, reset_and_seed
from tests.knowledge.fakes import HashingEmbeddingProvider

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
    "knowledge_chunk_embeddings",  # Step 8
    "action_requests",  # Step 10 (0004)
    "store_credit_transactions",  # Step 10 (0004)
    "agent_runs",  # Step 10 (0005)
    "audit_events",  # Step 10 (0005)
    "tenant_memberships",  # Phase 7 (0006)
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
        assert exts == {"plpgsql", "vector"}  # vector: Step 8 (kept on downgrade, see 0003)
    finally:
        # Leave the shared test database migrated and seeded for other tests.
        reset_and_seed(db_engine, cfg)


KNOWLEDGE_TABLES = {"knowledge_documents", "knowledge_chunks", "knowledge_chunk_embeddings"}
ACTION_TABLES = {
    "action_requests",
    "store_credit_transactions",
    "agent_runs",
    "audit_events",
    "tenant_memberships",
}
ECOMMERCE_TABLES = sorted(DOMAIN_TABLES - KNOWLEDGE_TABLES - ACTION_TABLES)


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
        command.upgrade(cfg, "0002")
        assert _ecommerce_snapshot(db_engine) == before
        with db_engine.connect() as conn:
            cols = set(
                conn.exec_driver_sql(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'knowledge_chunks'"
                ).scalars()
            )
        assert not {"embedding", "vector"} & cols  # chunks never carry vectors themselves
    finally:
        reset_and_seed(db_engine, cfg)


def _knowledge_snapshot(engine) -> dict[str, list[tuple]]:
    with engine.connect() as conn:
        return {
            t: [tuple(r) for r in conn.exec_driver_sql(f"SELECT * FROM {t} ORDER BY id")]  # noqa: S608
            for t in ("knowledge_documents", "knowledge_chunks")
        }


def _extensions(engine) -> set[str]:
    with engine.connect() as conn:
        return set(conn.exec_driver_sql("SELECT extname FROM pg_extension").scalars())


def test_embedding_migration_round_trip_keeps_the_vector_extension(db_engine, database_url):
    """0003 downgrade removes the embedding table and its constraints but intentionally
    keeps the (possibly shared) vector extension; upgrade creates it when missing."""
    cfg = alembic_config(database_url)
    try:
        with Session(db_engine) as session, session.begin():
            ingest_policies(session, DEFAULT_POLICY_DIR, ChunkingConfig())
            materialize_embeddings(session, HashingEmbeddingProvider(), batch_size=16)
        ecommerce, knowledge = _ecommerce_snapshot(db_engine), _knowledge_snapshot(db_engine)
        assert len(knowledge["knowledge_chunks"]) == 51

        command.downgrade(cfg, "0002")
        assert "knowledge_chunk_embeddings" not in _tables(db_engine)
        assert "vector" in _extensions(db_engine)  # asymmetric on purpose
        with db_engine.connect() as conn:
            uniques = set(
                conn.exec_driver_sql(
                    "SELECT conname FROM pg_constraint "
                    "WHERE conrelid = 'knowledge_chunks'::regclass"
                ).scalars()
            )
        assert "uq_knowledge_chunks_tenant_id_id" not in uniques
        assert _ecommerce_snapshot(db_engine) == ecommerce
        assert _knowledge_snapshot(db_engine) == knowledge

        with db_engine.begin() as conn:  # simulate a database that never had the extension
            conn.exec_driver_sql("DROP EXTENSION vector")
        command.upgrade(cfg, "head")
        assert "vector" in _extensions(db_engine)
        assert _ecommerce_snapshot(db_engine) == ecommerce
        assert _knowledge_snapshot(db_engine) == knowledge
        with db_engine.connect() as conn:
            assert (
                conn.exec_driver_sql("SELECT count(*) FROM knowledge_chunk_embeddings").scalar()
                == 0
            )
    finally:
        reset_and_seed(db_engine, cfg)


def test_action_migration_round_trip(db_engine, database_url):
    """0004 downgrade drops only the action tables (ledger first); re-upgrade restores
    them empty with their constraints; commerce and knowledge data are untouched."""
    cfg = alembic_config(database_url)
    try:
        ecommerce = _ecommerce_snapshot(db_engine)
        command.downgrade(cfg, "0003")
        tables = _tables(db_engine)
        assert "action_requests" not in tables and "store_credit_transactions" not in tables
        assert _ecommerce_snapshot(db_engine) == ecommerce
        command.upgrade(cfg, "head")
        with db_engine.connect() as conn:
            names = set(
                conn.exec_driver_sql(
                    "SELECT conname FROM pg_constraint "
                    "WHERE conrelid = 'store_credit_transactions'::regclass"
                ).scalars()
            )
            indexes = set(
                conn.exec_driver_sql(
                    "SELECT indexname FROM pg_indexes WHERE tablename = 'action_requests'"
                ).scalars()
            )
        assert {
            "uq_store_credit_transactions_tenant_id_idempotency_key",
            "ck_store_credit_transactions_amount_positive",
            "fk_store_credit_transactions_tenant_id_customer_id_customers",
        } <= names
        assert "uq_action_requests_open_target" in indexes
        assert _ecommerce_snapshot(db_engine) == ecommerce
    finally:
        reset_and_seed(db_engine, cfg)
