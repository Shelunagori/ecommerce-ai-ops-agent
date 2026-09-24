"""Developer-only: score a retriever on data/eval/policy_retrieval_cases.yaml (20 cases).

    uv run python -m scripts.eval_retrieval --retriever lexical
    uv run python -m scripts.eval_retrieval --retriever semantic \\
        --write data/eval/semantic_baseline_v1.json          # run on the Mac with Ollama

* Needs seeded tenants, ingested policies and (semantic) materialized embeddings.
* Prints metrics (document / exact-chunk hit@1 and hit@3 per tenant and overall) and a
  per-case lexical-vs-semantic comparison. --write stores a snapshot with provenance:
  retriever, provider/model/model digest/dimensions/input version (semantic), cases and
  corpus fingerprints, aggregate metrics and per-case ranks. No vectors or exact
  similarity scores are stored (they are environment dependent).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.db.session import read_only_session
from app.knowledge.chunking import CHUNKER
from app.knowledge.embeddings.errors import EmbeddingError
from app.knowledge.embeddings.provider import get_embedding_provider, resolve_profile
from app.knowledge.evaluation import (
    DEFAULT_CASES,
    cases_fingerprint,
    comparison,
    corpus_fingerprint,
    load_cases,
    metrics,
    per_case,
    run_cases,
)
from app.knowledge.retrieval import LexicalPolicyRetriever
from app.knowledge.semantic import SemanticKnowledgeRetriever
from scripts.run_assistant import _configure_logging
from scripts.seed_demo import DEMO_TENANTS, tenant_id_for

K = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_retrieval")
    parser.add_argument("--retriever", choices=("lexical", "semantic"), required=True)
    parser.add_argument("--write", type=Path, default=None, help="snapshot path (JSON)")
    ns = parser.parse_args(argv)
    _configure_logging()
    cases = load_cases()
    tenants = {t.slug: tenant_id_for(t.slug) for t in DEMO_TENANTS}
    snapshot: dict[str, object] = {
        "cases_file": DEFAULT_CASES.relative_to(DEFAULT_CASES.parents[2]).as_posix(),
        "cases_sha256": cases_fingerprint(),
        "chunker": CHUNKER,
        "k": K,
    }
    try:
        if ns.retriever == "semantic":
            provider = get_embedding_provider()
            profile = resolve_profile(provider)
            retriever = SemanticKnowledgeRetriever(read_only_session, provider)
            snapshot["embedding_profile"] = profile.as_dict()
        else:
            retriever = LexicalPolicyRetriever(read_only_session)
        outcomes = run_cases(retriever, cases, tenants, k=K)
        lexical = (
            outcomes
            if ns.retriever == "lexical"
            else run_cases(LexicalPolicyRetriever(read_only_session), cases, tenants, k=K)
        )
    except EmbeddingError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": exc.message}}), file=sys.stderr)
        return 1
    with read_only_session() as session:
        snapshot["corpus_sha256"] = corpus_fingerprint(session)
    snapshot = {
        "retriever": retriever.name,
        **snapshot,
        "metrics": metrics(outcomes),
        "cases": per_case(outcomes),
    }
    report = dict(snapshot)
    if ns.retriever == "semantic":
        report["comparison_vs_lexical"] = comparison(per_case(lexical), per_case(outcomes))
        report["lexical_metrics"] = metrics(lexical)
    print(json.dumps(report, indent=2))
    if ns.write:
        ns.write.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        print(f"snapshot written: {ns.write}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
