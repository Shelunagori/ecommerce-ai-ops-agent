"""Create / migrate the LangGraph PostgreSQL checkpoint tables (idempotent).

    uv run python -m scripts.setup_checkpoints          # uses DATABASE_URL

Run once per deployment after ``alembic upgrade head`` (docs/DEPLOYMENT.md). The tables
belong to ``langgraph-checkpoint-postgres``, which versions its own schema
(``checkpoint_migrations``); they hold only graph state, never secrets.
"""

import sys

from app.agent.graph.checkpoint import durable_checkpointer
from app.core.config import get_settings


def main() -> int:
    url = get_settings().database_url
    if not url:
        print("DATABASE_URL is not set.", file=sys.stderr)
        return 2
    cp = durable_checkpointer(url, min_size=1, max_size=1)
    try:
        cp.setup()
    finally:
        cp.close()
    print("checkpoint tables ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
