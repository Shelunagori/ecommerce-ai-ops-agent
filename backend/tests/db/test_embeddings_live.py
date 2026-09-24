"""OPT-IN live embedding tests against local Ollama (nomic-embed-text-v2-moe) + pgvector.

    ollama pull nomic-embed-text-v2-moe
    RUN_OLLAMA_INTEGRATION=1 TEST_DATABASE_URL=... uv run pytest tests/db/test_embeddings_live.py

Checks integration, not exact numbers: the digest resolves, real document and query vectors
have the expected size and finite non-zero values, EVERY current policy chunk is accepted
by the model (inputs are never truncated), and semantic retrieval runs against real
pgvector rows. If ``data/eval/semantic_baseline_v1.json`` exists and was recorded with the
same model digest, per-case ranks must match it (no similarity values are compared).
"""

import json
import math
import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest

from app.agent.context import AgentContext
from app.knowledge.chunking import ChunkingConfig
from app.knowledge.embeddings.inputs import document_input
from app.knowledge.embeddings.materialize import materialize_embeddings
from app.knowledge.embeddings.provider import (
    embed_document_inputs,
    embed_query_text,
    get_embedding_provider,
    resolve_profile,
)
from app.knowledge.evaluation import corpus_fingerprint, load_cases, metrics, per_case, run_cases
from app.knowledge.ingest import ingest_policies
from app.knowledge.semantic import SemanticKnowledgeRetriever
from app.knowledge.sources import DEFAULT_POLICY_DIR
from scripts.seed_demo import BLUEPEAK, NORTHSTAR, tenant_id_for

pytestmark = [
    pytest.mark.llm_integration,
    pytest.mark.skipif(
        os.getenv("RUN_OLLAMA_INTEGRATION") != "1", reason="set RUN_OLLAMA_INTEGRATION=1"
    ),
]
BASELINE = Path(__file__).resolve().parents[2] / "data" / "eval" / "semantic_baseline_v1.json"


@pytest.fixture(scope="module")
def provider():
    return get_embedding_provider()


def _finite_nonzero(v):
    return all(math.isfinite(x) for x in v) and any(x != 0 for x in v)


def test_live_digest_and_vectors(provider):
    profile = resolve_profile(provider)
    assert len(profile.model_digest) == 64 and profile.dimensions == 768
    [doc] = embed_document_inputs(
        provider,
        [
            document_input(
                "Refund Policy",
                "Refund Policy > Refund timing",
                "Fictional refunds are issued within 5 business days.",
            )
        ],
    )
    query = embed_query_text(provider, "When is a refund issued?")
    assert len(doc) == len(query) == 768
    assert _finite_nonzero(doc) and _finite_nonzero(query)


def test_live_materialization_accepts_the_whole_corpus_and_retrieval_runs(db_session, provider):
    ingest_policies(db_session, DEFAULT_POLICY_DIR, ChunkingConfig())
    report = materialize_embeddings(db_session, provider, batch_size=16)
    assert report.total_chunks == 51 and report.embedded + report.unchanged == 51

    @contextmanager
    def scope():
        yield db_session

    retriever = SemanticKnowledgeRetriever(scope, provider)
    ctx = AgentContext(tenant_id_for(BLUEPEAK.slug), "live-embed")
    result = retriever.retrieve("express delivery compensation", ctx, as_of=date(2026, 9, 1))
    assert result.results and all(-1.0 <= r.score <= 1.0 for r in result.results)
    assert result.embedding_profile == report.profile.key

    if BASELINE.exists():
        recorded = json.loads(BASELINE.read_text())
        if recorded["embedding_profile"]["model_digest"] != report.profile.model_digest:
            pytest.skip("semantic baseline was recorded with another model digest")
        assert recorded["corpus_sha256"] == corpus_fingerprint(db_session)
        tenants = {t.slug: tenant_id_for(t.slug) for t in (NORTHSTAR, BLUEPEAK)}
        outcomes = run_cases(retriever, load_cases(), tenants, k=recorded["k"])
        assert per_case(outcomes) == recorded["cases"]
        assert metrics(outcomes) == recorded["metrics"]
