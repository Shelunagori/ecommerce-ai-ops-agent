"""Developer-only: run the lexical baseline retriever for one tenant (no LLM, no embeddings).

    uv run python -m scripts.search_policies --tenant <uuid> \\
        --query "compensation for delayed shipment" --as-of 2026-09-01 [--limit 5]

* --tenant is TRUSTED context, validated against the tenants table first; the query text
  never selects a tenant, version or limit beyond the hard maximum.
* --as-of: YYYY-MM-DD, or an ISO datetime WITH timezone (naive datetimes are rejected).
  Omitted: today's UTC date.
* stdout: JSON (query, retriever, as_of, ranked chunks with citation, score, version and
  effective dates). Citations are tenant-relative; no tenant id or file path is printed.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import date, datetime

from app.agent.context import create_agent_context
from app.core.errors import TenantNotFoundError
from app.db.session import read_only_session
from app.knowledge.retrieval import DEFAULT_LIMIT, MAX_LIMIT, LexicalPolicyRetriever
from scripts.run_assistant import _configure_logging, _fail


def _as_of(value: str) -> date | datetime:
    if len(value) == 10:
        return date.fromisoformat(value)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime needs a timezone, e.g. 2026-09-01T12:00Z")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="search_policies", description=__doc__.split("\n")[0])
    parser.add_argument("--tenant", type=uuid.UUID, required=True, help="tenant UUID (trusted)")
    parser.add_argument("--query", required=True, help="search text (untrusted)")
    parser.add_argument("--as-of", type=_as_of, default=None, help="YYYY-MM-DD or tz datetime")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        choices=range(1, MAX_LIMIT + 1),
        metavar=f"1-{MAX_LIMIT}",
    )
    parser.add_argument("--request-id", default=None)
    ns = parser.parse_args(argv)
    _configure_logging()
    try:
        with read_only_session() as session:
            context = create_agent_context(session, ns.tenant, ns.request_id)
    except TenantNotFoundError:
        return _fail({"error": {"code": "tenant_not_found", "message": "Unknown tenant."}})
    except ValueError as exc:
        return _fail({"error": {"code": "invalid_request_id", "message": str(exc)}})

    result = LexicalPolicyRetriever(read_only_session).retrieve(
        ns.query, context, as_of=ns.as_of, limit=ns.limit
    )
    out = {"query": ns.query, **result.model_dump(mode="json")}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
