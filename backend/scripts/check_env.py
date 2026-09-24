"""Validate the environment for the configured APP_ENV (no network, no secret values printed).

    uv run python -m scripts.check_env            # exit 0 = ok, 1 = problems listed

Run as the first pre-deploy step (docs/DEPLOYMENT.md). Outside ``APP_ENV=production`` it
always passes; in production it lists every unmet requirement by setting NAME only.
"""

import sys

from pydantic import ValidationError

from app.core.config import Settings
from app.core.production import configuration_problems


def main(argv: list[str] | None = None) -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]).upper() for e in exc.errors()})
        print("invalid settings: " + ", ".join(fields), file=sys.stderr)
        return 1
    problems = configuration_problems(settings)
    if problems:
        print(f"APP_ENV={settings.app_env}: configuration problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"APP_ENV={settings.app_env}: configuration ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
