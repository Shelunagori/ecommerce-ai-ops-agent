"""Developer-only: ingest the SYNTHETIC policy corpus into knowledge tables (idempotent).

Usage (from backend/, after `scripts.seed_demo`):  uv run python -m scripts.ingest_policies

* Refuses to run when APP_ENV=production (no broad production mutation path).
* Reads only backend/data/policies/<tenant-slug>/<document_key>.v<N>.md.
* Atomic: validates everything first; any conflict aborts the run with no changes.
  - source_conflict:   an ingested version's file changed (versions are immutable)
  - chunking_mismatch: same source, different chunker/settings than the stored chunks
* Never deletes or updates rows. Re-running with identical sources changes nothing.
* stdout: JSON summary (per document: tenant slug, key, version, outcome, chunk count).
"""

from __future__ import annotations

import json
import logging
import sys

from app.core.config import get_settings
from app.core.logging import JsonFormatter
from app.knowledge.chunking import ChunkingConfig
from app.knowledge.ingest import IngestionConflictError, IngestionReport, ingest_policies
from app.knowledge.sources import DEFAULT_POLICY_DIR, PolicySourceError


def _summary(report: IngestionReport, status: str) -> dict:
    return {
        "status": status,
        "chunker": report.chunker,
        "chunking_hash": report.chunking_hash,
        "counts": {
            o: report.count(o)
            for o in ("inserted", "unchanged", "retired", "source_conflict", "chunking_mismatch")
        },
        "documents": [
            {
                "tenant": d.tenant,
                "document_key": d.document_key,
                "version": d.version,
                "outcome": d.outcome,
                "chunk_count": d.chunk_count,
            }
            for d in report.documents
        ],
    }


def main(argv: list[str] | None = None) -> int:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logging.getLogger().handlers[:] = [handler]
    logging.getLogger().setLevel(logging.INFO)
    if get_settings().app_env == "production":
        print("Refusing to ingest: APP_ENV=production.", file=sys.stderr)
        return 2
    from app.db.session import unit_of_work  # noqa: PLC0415 - needs DATABASE_URL only here

    try:
        with unit_of_work() as session:
            report = ingest_policies(session, DEFAULT_POLICY_DIR, ChunkingConfig.from_settings())
    except IngestionConflictError as exc:
        print(json.dumps(_summary(exc.report, "refused"), indent=2), file=sys.stderr)
        return 1
    except PolicySourceError as exc:
        print(
            json.dumps({"status": "invalid_source", "source": exc.source, "reason": exc.reason}),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(_summary(report, "ok"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
