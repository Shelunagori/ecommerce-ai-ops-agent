"""Developer-only: materialize policy-chunk embeddings for the configured profile.

Usage (from backend/, after `scripts.ingest_policies`, with `ollama pull
nomic-embed-text-v2-moe` done):   uv run python -m scripts.embed_policies

* Refuses to run when APP_ENV=production (no broad production mutation path).
* Resolves the concrete profile (provider, model tag, RESOLVED model digest, dimensions,
  input version) once, embeds only chunks missing for that profile, validates every vector
  and writes all rows in one transaction. Safe to rerun; other profiles are never touched.
* A stored row of the same profile with a different input hash is a stale conflict: the
  run aborts before any embedding call and writes nothing.
* stdout: JSON summary. stderr: safe logs / error envelope. Never prints vectors or text.
"""

from __future__ import annotations

import json
import sys

from app.core.config import get_settings
from app.knowledge.embeddings.errors import EmbeddingError, EmbeddingStaleConflictError
from app.knowledge.embeddings.materialize import materialize_embeddings
from app.knowledge.embeddings.provider import get_embedding_provider
from scripts.run_assistant import _configure_logging


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    settings = get_settings()
    if settings.app_env == "production":
        print("Refusing to embed: APP_ENV=production.", file=sys.stderr)
        return 2
    from app.db.session import unit_of_work  # noqa: PLC0415 - needs DATABASE_URL only here

    provider = get_embedding_provider(settings)
    try:
        with unit_of_work() as session:
            report = materialize_embeddings(
                session, provider, batch_size=settings.embedding_batch_size
            )
    except EmbeddingStaleConflictError as exc:
        body = {"status": "refused", "error": {"code": exc.code, "message": exc.message}}
        body.update(exc.report.as_dict())
        print(json.dumps(body, indent=2), file=sys.stderr)
        return 1
    except EmbeddingError as exc:
        print(
            json.dumps({"status": "failed", "error": {"code": exc.code, "message": exc.message}}),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"status": "ok", **report.as_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
