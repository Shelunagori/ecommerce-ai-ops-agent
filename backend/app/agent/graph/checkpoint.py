"""Checkpointer factories (the only LangGraph-checkpoint imports outside the graph nodes).

* ``ephemeral_checkpointer()`` - ``InMemorySaver``: unit tests, RAG evaluation, CLI threads.
* ``durable_checkpointer(database_url)`` - the OFFICIAL ``langgraph-checkpoint-postgres``
  ``PostgresSaver`` on a psycopg ``ConnectionPool`` (autocommit, prepare_threshold=0 so it
  also works behind transaction poolers, dict rows - as the package requires). Production
  graphs use it: approval interrupts survive process restarts and can be resumed from any
  process that can reach the database.

Serialisation: LangGraph's ``JsonPlusSerializer`` with ``pickle_fallback=False`` and strict
msgpack (``allowed_msgpack_modules=None``): no pickle, and no arbitrary class is revived from
a checkpoint. Graph state therefore holds only messages and JSON primitives.

Schema: the saver's own tables (``checkpoints``, ``checkpoint_blobs``, ``checkpoint_writes``,
``checkpoint_migrations``) are created/upgraded by its idempotent ``setup()`` - run
``python -m scripts.setup_checkpoints`` once per deployment (see docs/DEPLOYMENT.md). They
are not Alembic-managed because the package versions its own schema.

Retention: nothing expires automatically. ``delete_checkpoint_thread`` removes one
tenant-scoped thread (all checkpoints and writes); see docs for the retention policy.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


def ephemeral_checkpointer() -> InMemorySaver:
    return InMemorySaver()


def libpq_url(database_url: str) -> str:
    """SQLAlchemy URL (``postgresql+psycopg://``) -> libpq conninfo URL (``postgresql://``)."""
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://", "postgres://"):
        if database_url.startswith(prefix):
            return "postgresql://" + database_url[len(prefix) :]
    return database_url


def safe_serde() -> JsonPlusSerializer:
    """No pickle, and STRICT msgpack: only LangGraph's built-in safe types (LangChain messages,
    primitives, ...) are revived; any other class in a checkpoint is refused, not imported."""
    return JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=None)


class DurableCheckpointer:
    """Owns the connection pool; ``saver`` is the ``PostgresSaver`` given to the graph."""

    def __init__(self, database_url: str, *, min_size: int = 1, max_size: int = 5) -> None:
        self.pool = ConnectionPool(
            libpq_url(database_url),
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=True,
        )
        self.saver = PostgresSaver(self.pool, serde=safe_serde())  # type: ignore[arg-type]

    def setup(self) -> None:
        """Create / migrate the checkpoint tables (idempotent)."""
        self.saver.setup()

    def close(self) -> None:
        self.pool.close()


def durable_checkpointer(database_url: str, **kw: int) -> DurableCheckpointer:
    return DurableCheckpointer(database_url, **kw)


def delete_checkpoint_thread(checkpointer: object, thread_key: str) -> None:
    """Delete every checkpoint of one internal (tenant-derived) thread key."""
    delete = getattr(checkpointer, "delete_thread", None)
    if delete is None:  # pragma: no cover - both savers implement it
        raise TypeError("checkpointer cannot delete threads")
    delete(thread_key)
