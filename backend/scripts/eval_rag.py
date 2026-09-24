"""Developer-only: LIVE RAG/agent measurement on data/eval/rag_agent_cases.yaml (Step 9).

    uv run python -m scripts.embed_policies            # current embedding profile materialized
    uv run python -m scripts.eval_rag --provider ollama [--model qwen3:4b-instruct] \\
        --write data/eval/rag_live_measurement_v1.json

* Runs every case through the production LangGraph assistant (prompt v2, semantic policy
  retrieval, real commerce tools on read-only sessions) with the REAL chat model. The
  retriever uses each case's fixed ``as_of`` as its clock (no wall-clock dependence).
* Output: structured per-case outcomes and aggregate metrics - never answer prose, prompts,
  retrieved content, query text or vectors. ``--write`` stores a versioned MEASUREMENT with
  provenance (cases/corpus fingerprints, prompt version, LLM provider/model/digest where
  available, embedding profile, UTC timestamp). Generative behaviour drifts; this file is a
  measurement, not a CI regression snapshot. CI uses scripted models (tests/db/test_rag_eval.py).
* Synthetic demo data only. Refuses APP_ENV=production.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from app.agent.graph.profile import RAG_PROFILE
from app.agent.llm import LLMError, get_llm_provider
from app.agent.llm.config import SUPPORTED_PROVIDERS
from app.agent.rag.evaluation import (
    DEFAULT_RAG_CASES,
    load_rag_cases,
    rag_cases_fingerprint,
    rag_metrics,
    run_rag_cases,
)
from app.agent.tools import build_commerce_tools
from app.core.config import get_settings
from app.db.session import read_only_session
from app.knowledge.embeddings.errors import EmbeddingError
from app.knowledge.embeddings.provider import (
    get_embedding_provider,
    resolve_ollama_model_digest,
    resolve_profile,
)
from app.knowledge.evaluation import corpus_fingerprint
from app.knowledge.semantic import SemanticKnowledgeRetriever
from scripts.run_assistant import _configure_logging
from scripts.seed_demo import DEMO_TENANTS, tenant_id_for


def _noon(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 12, tzinfo=UTC)


def _ollama_model_digest(model: str) -> str | None:
    """Resolved digest of the chat model (best effort; None when unavailable)."""
    try:
        return resolve_ollama_model_digest(model, base_url=get_settings().ollama_base_url)
    except Exception:  # noqa: BLE001 - provenance is best effort, never fatal
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_rag")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="ollama")
    parser.add_argument("--model", default=None, help="override the configured chat model")
    parser.add_argument("--write", type=Path, default=None, help="measurement path (JSON)")
    ns = parser.parse_args(argv)
    settings = get_settings()
    if settings.app_env == "production":
        print("Refusing to evaluate: APP_ENV=production.", file=sys.stderr)
        return 2
    _configure_logging()
    cases = load_rag_cases()
    tenants = {t.slug: tenant_id_for(t.slug) for t in DEMO_TENANTS}
    try:
        llm = get_llm_provider(provider=ns.provider, model=ns.model)
        embeddings = get_embedding_provider()
        profile = resolve_profile(embeddings)
    except (LLMError, EmbeddingError) as exc:
        print(json.dumps({"error": {"code": exc.code, "message": exc.message}}), file=sys.stderr)
        return 1

    def retriever_for(day: date) -> SemanticKnowledgeRetriever:
        return SemanticKnowledgeRetriever(read_only_session, embeddings, clock=lambda: _noon(day))

    outcomes = run_rag_cases(cases, lambda _c: llm, retriever_for, build_commerce_tools(), tenants)
    with read_only_session() as session:
        corpus = corpus_fingerprint(session)
    measurement = {
        "kind": "rag-live-measurement",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cases_file": DEFAULT_RAG_CASES.relative_to(DEFAULT_RAG_CASES.parents[2]).as_posix(),
        "cases_sha256": rag_cases_fingerprint(),
        "corpus_sha256": corpus,
        "prompt_version": RAG_PROFILE.prompt_version,
        "llm": {
            "provider": llm.info.provider,
            "model": llm.info.model,
            "model_digest": _ollama_model_digest(llm.info.model)
            if llm.info.provider == "ollama"
            else None,
        },
        "embedding_profile": profile.as_dict(),
        "metrics": rag_metrics(outcomes),
        "cases": outcomes,
    }
    print(json.dumps(measurement, indent=2))
    if ns.write:
        ns.write.write_text(json.dumps(measurement, indent=2) + "\n", encoding="utf-8")
        print(f"measurement written: {ns.write}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
