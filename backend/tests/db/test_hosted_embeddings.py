"""Hosted (Gemini) embedding profile on real pgvector: same materialisation and retrieval
contract, separate embedding space, never mixed with the local Ollama profile. The Gemini
SDK is faked; an opt-in live test runs only with RUN_GEMINI_INTEGRATION=1 + GEMINI_API_KEY."""

import os
from contextlib import contextmanager
from datetime import date

import pytest
from sqlalchemy import func, select

from app.agent.context import AgentContext
from app.core.config import get_settings
from app.knowledge.chunking import ChunkingConfig
from app.knowledge.embeddings.errors import EmbeddingProfileNotMaterializedError
from app.knowledge.embeddings.gemini import GeminiEmbeddingProvider
from app.knowledge.embeddings.materialize import materialize_embeddings
from app.knowledge.embeddings.provider import embed_query_text, resolve_profile
from app.knowledge.ingest import ingest_policies
from app.knowledge.semantic import SemanticKnowledgeRetriever
from app.knowledge.sources import DEFAULT_POLICY_DIR
from app.models import KnowledgeChunkEmbedding
from tests.knowledge.fakes import HashingEmbeddingProvider
from tests.knowledge.test_gemini_embeddings import FakeModels, provider

SEPT = date(2026, 9, 1)


@pytest.fixture
def kb(db_session):
    ingest_policies(db_session, DEFAULT_POLICY_DIR, ChunkingConfig())
    return db_session


@pytest.fixture
def scope(kb):
    @contextmanager
    def _scope():
        yield kb

    return _scope


def by_provider(session):
    return dict(
        session.execute(
            select(KnowledgeChunkEmbedding.provider, func.count()).group_by(
                KnowledgeChunkEmbedding.provider
            )
        ).all()
    )


def test_hosted_profile_materialises_separately_and_retrieves_only_its_own_rows(
    kb, scope, tenant_b
):
    materialize_embeddings(kb, HashingEmbeddingProvider(), batch_size=16)  # local profile
    gemini, models = provider()
    ctx = AgentContext(tenant_b.tenant_id, "hosted")
    with pytest.raises(EmbeddingProfileNotMaterializedError):  # never falls back to Ollama rows
        SemanticKnowledgeRetriever(scope, gemini).retrieve("delayed shipment", ctx, as_of=SEPT)
    report = materialize_embeddings(kb, gemini, batch_size=16)
    assert (report.embedded, report.profile.input_version) == (
        51,
        "policy-embedding-input-gemini-v1",
    )
    assert by_provider(kb) == {"ollama": 51, "gemini": 51}
    sent = [c.parts[0].text for _m, contents, _cfg in models.calls for c in contents]
    assert all(t.startswith("title: ") and " | text: Section: " in t for t in sent)
    result = SemanticKnowledgeRetriever(scope, gemini).retrieve(
        "delayed shipment compensation", ctx, as_of=SEPT, limit=3
    )
    assert result.embedding_profile == resolve_profile(gemini).key and len(result.results) == 3
    assert materialize_embeddings(kb, gemini, batch_size=16).embedded == 0  # idempotent


def test_a_new_hosted_model_revision_is_a_new_profile(kb, scope, tenant_b):
    first, _ = provider()
    materialize_embeddings(kb, first, batch_size=16)

    class NewRevision(FakeModels):
        def get(self, *, model):
            return type("M", (), {"name": f"models/{model}", "version": "3"})()

    second, _ = provider(NewRevision())
    assert resolve_profile(second).key != resolve_profile(first).key
    with pytest.raises(EmbeddingProfileNotMaterializedError):
        SemanticKnowledgeRetriever(scope, second).retrieve(
            "refund", AgentContext(tenant_b.tenant_id), as_of=SEPT
        )


@pytest.mark.llm_integration
@pytest.mark.skipif(
    os.getenv("RUN_GEMINI_INTEGRATION") != "1" or not os.getenv("GEMINI_API_KEY"),
    reason="set RUN_GEMINI_INTEGRATION=1 and GEMINI_API_KEY (synthetic data only)",
)
def test_live_gemini_embeddings():
    get_settings.cache_clear()
    settings = get_settings()
    p = GeminiEmbeddingProvider.from_settings(settings)
    profile = resolve_profile(p)
    vector = embed_query_text(p, "How long does a refund take?")
    assert len(vector) == settings.embedding_dimensions and profile.provider == "gemini"
