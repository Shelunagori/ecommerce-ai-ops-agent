"""Pre-deploy step (Railway ``preDeployCommand``; any host's release phase):

    python -m scripts.predeploy

1. ``check_env``          - refuse to release an unsafe production configuration;
2. ``alembic upgrade head`` - apply schema migrations (forward only; see DEPLOYMENT.md);
3. ``setup_checkpoints``  - create / migrate the LangGraph checkpoint tables.

Stops at the first failure with a non-zero exit code so the release is aborted and the
previous deployment keeps serving. Idempotent: safe to re-run. Never prints secrets.
"""

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config

from scripts import check_env, setup_checkpoints

BACKEND_DIR = Path(__file__).resolve().parents[1]


def upgrade_head() -> int:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    try:
        command.upgrade(cfg, "head")
    except Exception as exc:  # noqa: BLE001 - report the type only (URLs can hold passwords)
        print(f"migration failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print("migrations at head")
    return 0


STEPS = (
    ("check_env", lambda: check_env.main([])),
    ("migrations", upgrade_head),
    ("checkpoints", setup_checkpoints.main),
)


def main(argv: list[str] | None = None) -> int:
    for name, step in STEPS:
        code = step()
        if code != 0:
            print(f"pre-deploy aborted at step: {name}", file=sys.stderr)
            return code
    print("pre-deploy complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
