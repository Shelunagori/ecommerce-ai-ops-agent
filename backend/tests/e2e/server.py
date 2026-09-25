"""Local end-to-end harness for the browser tests in ``frontend/e2e`` (Phase 10).

The REAL FastAPI app on a REAL PostgreSQL database: migrations, synthetic seed, policy
ingestion, commerce tools, lexical policy retrieval (no embedding model needed), grounding,
the action service, durable PostgreSQL checkpoints and run records. Only the chat MODEL is
the deterministic ``KeywordChatModel`` (no LLM, no network). Demo auth mode.

Test-only and destructive: it RESETS the database, so it refuses anything but a dedicated
database whose name ends in ``_test`` (the same guard as the pytest suite) and refuses
``APP_ENV=production``. Do not run it while the pytest DB suite is running.

    TEST_DATABASE_URL=postgresql://…/commerceops_test \
      uv run python -m tests.e2e.server --port 8100 --origin http://127.0.0.1:3100

``POST /__e2e/reset`` (only on this harness app) restores the pristine seed between tests.
``POST /__e2e/pacing`` ``{"delay_ms": n}`` makes each model call, commerce tool, policy
retrieval and approved action execution take at least ``n`` ms, so the browser can observe
every live step while it runs (the steps themselves are still the real ones).
``POST /__e2e/faults``
``{"retrieval": true}`` makes policy retrieval fail (controlled failure injection). Both are
reset by ``/__e2e/reset``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, make_url, text
from sqlalchemy.orm import Session

BACKEND_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_TABLES = (
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoints",
    "checkpoint_migrations",
)
ACTION_TABLES = (
    "public_demo_usage",
    "audit_events",
    "agent_runs",
    "store_credit_transactions",
    "action_requests",
)


# Test-only knobs for the live-trace browser tests (see the module docstring).
CONTROL: dict[str, float | bool] = {"delay_ms": 0.0, "retrieval_fault": False}


def pace() -> None:
    delay = float(CONTROL["delay_ms"])
    if delay > 0:
        time.sleep(delay / 1000)


class PacedRetriever:
    """Wraps the real lexical retriever: optional pacing and failure injection."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def retrieve(self, query, context, *, as_of=None, limit=5):
        pace()
        if CONTROL["retrieval_fault"]:
            raise RuntimeError("e2e injected retrieval fault")
        return self._inner.retrieve(query, context, as_of=as_of, limit=limit)


def install_pacing() -> None:
    """Pace the real model and tool calls in THIS test process only."""
    from app.actions.service import ActionService
    from app.agent.assistant.executor import ToolExecutor
    from tests.e2e.keyword_model import KeywordChatModel

    tool_execute = ToolExecutor.execute
    model_invoke = KeywordChatModel.invoke
    action_execute = ActionService.execute

    def paced_action(self, *args, **kwargs):
        pace()
        return action_execute(self, *args, **kwargs)

    def paced_execute(self, *args, **kwargs):
        pace()
        return tool_execute(self, *args, **kwargs)

    def paced_invoke(self, messages):
        pace()
        return model_invoke(self, messages)

    ToolExecutor.execute = paced_execute  # type: ignore[method-assign]
    ActionService.execute = paced_action  # type: ignore[method-assign]
    KeywordChatModel.invoke = paced_invoke  # type: ignore[method-assign]


def database_url() -> str:
    from app.db.url import normalize_database_url

    raw = os.getenv("TEST_DATABASE_URL")
    if not raw:
        sys.exit("TEST_DATABASE_URL is required (a dedicated database ending in '_test')")
    url = normalize_database_url(raw)
    if not (make_url(url).database or "").endswith("_test"):
        sys.exit("Refusing to run: TEST_DATABASE_URL must name a database ending in '_test'")
    if os.getenv("APP_ENV") == "production":
        sys.exit("Refusing to run the e2e harness with APP_ENV=production")
    return url


def drop_checkpoint_tables(engine) -> None:
    with engine.begin() as conn:
        for table in CHECKPOINT_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))  # noqa: S608 - fixed names


def prepare(url: str):
    from alembic import command
    from alembic.config import Config

    from app.knowledge.chunking import ChunkingConfig
    from app.knowledge.ingest import ingest_policies
    from app.knowledge.sources import DEFAULT_POLICY_DIR
    from scripts.seed_demo import seed

    engine = create_engine(url, future=True)
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    drop_checkpoint_tables(engine)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    with Session(engine) as session, session.begin():
        seed(session)
        ingest_policies(session, DEFAULT_POLICY_DIR, ChunkingConfig())
    with engine.connect() as conn:
        statuses = conn.execute(text("SELECT id, status FROM orders")).all()
    return engine, statuses


def build_app(url: str, origin: str):
    os.environ["DATABASE_URL"] = url  # read by app.db.session's lazily-built engine
    from app.core.config import get_settings

    get_settings.cache_clear()

    import app.db.session as db
    from app.actions.service import ActionService
    from app.agent.graph import CommerceGraphAssistant
    from app.agent.graph.checkpoint import durable_checkpointer
    from app.agent.graph.profile import AGENT_PROFILE
    from app.agent.llm.provider import ChatModelProvider, ProviderInfo, RetryPolicy
    from app.api.agent_runtime import AgentRuntime
    from app.core.config import Settings
    from app.knowledge.retrieval import LexicalPolicyRetriever
    from app.main import create_app
    from app.observability.runs import RunRecorder
    from tests.e2e.keyword_model import KeywordChatModel

    engine, statuses = prepare(url)
    install_pacing()
    checkpointer = durable_checkpointer(url)
    checkpointer.setup()
    actions = ActionService()
    assistant = CommerceGraphAssistant(
        ChatModelProvider(
            KeywordChatModel(),  # type: ignore[arg-type]
            ProviderInfo(provider="fake", model="e2e-keyword"),
            RetryPolicy(max_retries=0),
        ),
        checkpointer=checkpointer.saver,
        retriever=PacedRetriever(LexicalPolicyRetriever(db.read_only_session)),
        profile=AGENT_PROFILE,
        actions=actions,
        run_recorder=RunRecorder(),
    )

    def close() -> None:
        checkpointer.close()
        drop_checkpoint_tables(engine)  # leave the test DB as the pytest suite expects
        engine.dispose()

    settings = Settings(
        _env_file=None,
        app_env="test",
        auth_mode="demo",
        database_url=url,
        cors_origins=[origin],
    )
    app = create_app(settings)
    app.state.agent_runtime = AgentRuntime(assistant, actions, closer=close)

    @contextmanager
    def _writes():
        with engine.begin() as conn:
            yield conn

    @app.post("/__e2e/pacing", include_in_schema=False)
    def pacing(body: dict[str, float]) -> dict[str, float]:
        CONTROL["delay_ms"] = max(0.0, min(float(body.get("delay_ms", 0)), 5000.0))
        return {"delay_ms": float(CONTROL["delay_ms"])}

    @app.post("/__e2e/faults", include_in_schema=False)
    def faults(body: dict[str, bool]) -> dict[str, bool]:
        CONTROL["retrieval_fault"] = bool(body.get("retrieval", False))
        return {"retrieval": bool(CONTROL["retrieval_fault"])}

    @app.post("/__e2e/reset", include_in_schema=False)
    def reset() -> dict[str, bool]:
        CONTROL.update({"delay_ms": 0.0, "retrieval_fault": False})
        with _writes() as conn:
            for table in ACTION_TABLES:
                conn.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed names
            for table in CHECKPOINT_TABLES[:3]:
                conn.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed names
            for order_id, status in statuses:
                conn.execute(
                    text("UPDATE orders SET status = :s WHERE id = :i"),
                    {"s": status, "i": order_id},
                )
        return {"ok": True}

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--origin", default="http://127.0.0.1:3100")
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(build_app(database_url(), args.origin), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
