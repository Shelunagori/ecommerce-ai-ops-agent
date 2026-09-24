"""Semantic retrieval (semantic-pgvector-v1) on real PostgreSQL with deterministic vectors.

Document vectors come from the hashing fake; query vectors are chosen per test (e.g. the
exact stored vector of one chunk) so nearest-neighbour expectations are exact."""

import json
import logging
import math
from contextlib import contextmanager
from datetime import date, datetime

import pytest
from sqlalchemy import func, select

from app.agent.context import AgentContext
from app.knowledge.chunking import ChunkingConfig
from app.knowledge.embeddings.errors import (
    EmbeddingNonFiniteError,
    EmbeddingProfileNotMaterializedError,
    EmbeddingZeroVectorError,
)
from app.knowledge.embeddings.inputs import document_input
from app.knowledge.embeddings.materialize import materialize_embeddings
from app.knowledge.embeddings.provider import resolve_profile
from app.knowledge.ingest import ingest_policies
from app.knowledge.limits import MAX_LIMIT
from app.knowledge.retrieval import LexicalPolicyRetriever
from app.knowledge.semantic import SemanticKnowledgeRetriever
from app.knowledge.sources import DEFAULT_POLICY_DIR
from app.models import KnowledgeChunk, KnowledgeChunkEmbedding, KnowledgeDocument
from app.services import KnowledgeQueries
from tests.db.conftest import FIXED_NOW
from tests.knowledge.fakes import (
    DIGEST_A,
    DIGEST_B,
    BrokenProvider,
    HashingEmbeddingProvider,
    TableEmbeddingProvider,
    hashed_vector,
)

SEPT = date(2026, 9, 1)


@pytest.fixture
def kb(db_session):
    ingest_policies(db_session, DEFAULT_POLICY_DIR, ChunkingConfig())
    materialize_embeddings(db_session, HashingEmbeddingProvider(), batch_size=16)
    return db_session


@pytest.fixture
def scope(kb):
    @contextmanager
    def _scope():
        yield kb

    return _scope


@pytest.fixture
def ctx_a(tenant_a):
    return AgentContext(tenant_a.tenant_id, "req-a")


@pytest.fixture
def ctx_b(tenant_b):
    return AgentContext(tenant_b.tenant_id, "req-b")


def input_of(session, tenant_id, key, version, index) -> str:
    chunk, doc = session.execute(
        select(KnowledgeChunk, KnowledgeDocument)
        .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
        .where(
            KnowledgeChunk.tenant_id == tenant_id,
            KnowledgeDocument.document_key == key,
            KnowledgeDocument.version == version,
            KnowledgeChunk.chunk_index == index,
        )
    ).one()
    return document_input(doc.title, chunk.section, chunk.content)


def query_provider(query_vector, **kw) -> TableEmbeddingProvider:
    """Query 'x' embeds to ``query_vector``; everything else hashes like the stored docs."""
    return TableEmbeddingProvider(
        table={"search_query: x": query_vector},
        default=lambda t: hashed_vector(t, kw.get("dimensions", 768)),
        **kw,
    )


def retriever(scope, provider):
    return SemanticKnowledgeRetriever(scope, provider, clock=lambda: FIXED_NOW)


def cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# --- ranking ---------------------------------------------------------------------------------
def test_nearest_chunk_ranks_first_with_similarity_one(kb, scope, ctx_b):
    target = input_of(kb, ctx_b.tenant_id, "shipping-policy", 1, 2)
    provider = query_provider(hashed_vector(target, 768))
    result = retriever(scope, provider).retrieve("x", ctx_b, as_of=SEPT, limit=3)
    top = result.results[0]
    assert top.citation == "policy://shipping-policy/v1#chunk-2"
    assert top.score == pytest.approx(1.0, abs=1e-6) and top.rank == 1


def test_cosine_ordering_matches_an_independent_computation(kb, scope, ctx_a):
    q = hashed_vector("search_query: refund timing business days store credit", 768)
    provider = query_provider(q)
    result = retriever(scope, provider).retrieve("x", ctx_a, as_of=SEPT, limit=MAX_LIMIT)
    # Independent exact cosine over every ELIGIBLE Northstar chunk on SEPT.
    eligible = kb.execute(
        select(KnowledgeChunkEmbedding.embedding, KnowledgeChunk.id)
        .join(KnowledgeChunk, KnowledgeChunk.id == KnowledgeChunkEmbedding.chunk_id)
        .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
        .where(
            KnowledgeChunk.tenant_id == ctx_a.tenant_id,
            KnowledgeDocument.effective_from <= SEPT,
            (KnowledgeDocument.effective_to.is_(None)) | (KnowledgeDocument.effective_to > SEPT),
        )
    ).all()
    expected = sorted(((cos(v, q), cid) for v, cid in eligible), key=lambda t: -t[0])
    got = [(r.score, r.chunk_id) for r in result.results]
    # pgvector stores float4, so compare scores with a float4-level tolerance.
    assert [s for s, _ in got] == pytest.approx([s for s, _ in expected[:MAX_LIMIT]], abs=1e-4)
    assert [r.score for r in result.results] == sorted(
        [r.score for r in result.results], reverse=True
    )


def test_score_is_similarity_not_distance(kb, scope, ctx_b):
    target = input_of(kb, ctx_b.tenant_id, "returns-policy", 1, 1)
    result = retriever(scope, query_provider(hashed_vector(target, 768))).retrieve(
        "x", ctx_b, as_of=SEPT, limit=3
    )
    scores = [r.score for r in result.results]
    assert scores[0] == pytest.approx(1.0, abs=1e-6)  # identical vector -> 1, not 0
    assert all(-1.0 <= s <= 1.0 for s in scores) and scores == sorted(scores, reverse=True)
    assert result.score_type == "cosine_similarity"


@pytest.mark.parametrize("limit", [1, 3, MAX_LIMIT])
def test_limit_is_enforced(scope, ctx_a, limit):
    provider = query_provider(hashed_vector("search_query: shipping", 768))
    assert len(retriever(scope, provider).retrieve("x", ctx_a, limit=limit).results) == limit


@pytest.mark.parametrize("limit", [0, -1, MAX_LIMIT + 1, 10_000])
def test_limit_has_a_hard_maximum(scope, ctx_a, limit):
    provider = HashingEmbeddingProvider()
    with pytest.raises(ValueError):
        retriever(scope, provider).retrieve("refund", ctx_a, limit=limit)
    assert provider.query_calls == []


# --- eligibility: tenant, time, profile, dimensions ----------------------------------------
def test_tenant_isolation(kb, scope, ctx_a, ctx_b):
    b_target = input_of(kb, ctx_b.tenant_id, "shipping-policy", 1, 2)  # BluePeak Priority text
    provider = query_provider(hashed_vector(b_target, 768))
    result = retriever(scope, provider).retrieve("x", ctx_a, as_of=SEPT, limit=MAX_LIMIT)
    a_ids = set(
        kb.scalars(select(KnowledgeChunk.id).where(KnowledgeChunk.tenant_id == ctx_a.tenant_id))
    )
    assert result.results and {r.chunk_id for r in result.results} <= a_ids
    assert all("EUR" not in r.content and "Priority" not in r.content for r in result.results)


def test_temporal_version_is_part_of_eligibility(kb, scope, ctx_a):
    v2 = input_of(kb, ctx_a.tenant_id, "refund-policy", 2, 1)
    provider = query_provider(hashed_vector(v2, 768))
    june = retriever(scope, provider).retrieve("x", ctx_a, as_of=date(2026, 6, 10), limit=MAX_LIMIT)
    refunds = {r.version for r in june.results if r.document_key == "refund-policy"}
    assert refunds == {1}  # the (identical) v2 vector is not eligible in June
    sept = retriever(scope, provider).retrieve("x", ctx_a, as_of=SEPT, limit=1)
    assert sept.results[0].citation == "policy://refund-policy/v2#chunk-1"
    assert retriever(scope, provider).retrieve("x", ctx_a, as_of=date(2025, 12, 31)).results == []


def test_default_as_of_uses_the_injected_clock(scope, ctx_b):
    result = retriever(scope, HashingEmbeddingProvider()).retrieve("delayed compensation", ctx_b)
    assert result.as_of == FIXED_NOW.date()
    assert {
        r.version for r in result.results if r.document_key == "delayed-shipment-compensation"
    } <= {2}


def test_unmaterialized_digest_fails_clearly_without_fallback(scope, ctx_a):
    provider = HashingEmbeddingProvider(digest=DIGEST_B)  # vectors exist only for DIGEST_A
    with pytest.raises(EmbeddingProfileNotMaterializedError) as exc:
        retriever(scope, provider).retrieve("refund", ctx_a, as_of=SEPT)
    assert exc.value.code == "embedding_profile_not_materialized"
    assert provider.query_calls == []  # no query embedding for an unusable profile


def test_each_digest_only_searches_its_own_vectors(kb, scope, ctx_a):
    # Profile B: every document vector is the NEGATION of profile A's.
    class Negated(HashingEmbeddingProvider):
        def embed_documents(self, texts):
            return [[-x for x in v] for v in super().embed_documents(texts)]

    materialize_embeddings(kb, Negated(digest=DIGEST_B), batch_size=16)
    target = input_of(kb, ctx_a.tenant_id, "returns-policy", 1, 1)
    q = hashed_vector(target, 768)
    a = retriever(scope, query_provider(q, digest=DIGEST_A)).retrieve("x", ctx_a, as_of=SEPT)
    b = retriever(scope, query_provider(q, digest=DIGEST_B)).retrieve("x", ctx_a, as_of=SEPT)
    target_id = a.results[0].chunk_id
    assert a.results[0].score == pytest.approx(1.0, abs=1e-6)
    # In profile B the same chunk's vector is -q (similarity -1): it cannot be near the top.
    assert target_id not in [r.chunk_id for r in b.results]
    assert b.results[0].score < 1.0 - 1e-3
    assert DIGEST_A in a.embedding_profile and DIGEST_B in b.embedding_profile


def test_other_dimensions_never_mix(kb, scope, ctx_a):
    materialize_embeddings(kb, HashingEmbeddingProvider(dimensions=256), batch_size=16)
    r768 = retriever(scope, HashingEmbeddingProvider()).retrieve("refund", ctx_a, as_of=SEPT)
    r256 = retriever(scope, HashingEmbeddingProvider(dimensions=256)).retrieve(
        "refund", ctx_a, as_of=SEPT
    )
    assert r768.results and r256.results
    assert "/768/" in r768.embedding_profile and "/256/" in r256.embedding_profile
    profile = resolve_profile(HashingEmbeddingProvider())
    with pytest.raises(ValueError):
        KnowledgeQueries(kb, ctx_a.tenant).nearest_chunks([1.0] * 256, profile, SEPT, 5)


def test_other_input_versions_are_not_searched(kb, scope, ctx_a):
    kb.query(KnowledgeChunkEmbedding).update({"input_version": "policy-embedding-input-v0"})
    with pytest.raises(EmbeddingProfileNotMaterializedError):
        retriever(scope, HashingEmbeddingProvider()).retrieve("refund", ctx_a, as_of=SEPT)


# --- contract, citations, query handling ----------------------------------------------------------
def test_citations_and_result_shape_match_the_lexical_retriever(kb, scope, ctx_a):
    sem = retriever(scope, HashingEmbeddingProvider()).retrieve("PO boxes", ctx_a, as_of=SEPT)
    lex = LexicalPolicyRetriever(scope).retrieve("PO boxes", ctx_a, as_of=SEPT)
    assert sem.retriever == "semantic-pgvector-v1" and lex.retriever == "lexical-pg-fts-v1"
    assert set(sem.results[0].model_dump()) == set(lex.results[0].model_dump())
    by_id = {r.chunk_id: r.citation for r in lex.results}
    q = KnowledgeQueries(kb, ctx_a.tenant)
    for r in sem.results:
        assert r.citation == q.get_chunk(r.chunk_id).citation
        assert by_id.get(r.chunk_id, r.citation) == r.citation
    assert (sem.term_count, sem.lexeme_count) == (None, None)
    dumped = json.dumps(sem.model_dump(mode="json"))
    for leak in (str(ctx_a.tenant_id), ".md", "northstar-commerce"):
        assert leak not in dumped


def test_query_embedding_is_prefixed_and_never_stored(kb, scope, ctx_a):
    before = kb.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding))
    provider = HashingEmbeddingProvider()
    retriever(scope, provider).retrieve("  How long can I return an item?  ", ctx_a, as_of=SEPT)
    assert provider.query_calls == ["search_query: How long can I return an item?"]
    assert kb.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding)) == before


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_returns_empty_without_calling_the_provider(scope, ctx_a, query):
    provider = HashingEmbeddingProvider()
    result = retriever(scope, provider).retrieve(query, ctx_a, as_of=SEPT)
    assert result.results == [] and provider.query_calls == [] and provider.digest_calls == 0


def test_long_query_is_capped_at_500_chars(scope, ctx_a):
    provider = HashingEmbeddingProvider()
    retriever(scope, provider).retrieve("refund " * 200, ctx_a, as_of=SEPT)
    assert len(provider.query_calls[0]) == len("search_query: ") + 500


@pytest.mark.parametrize(
    ("vector", "error"),
    [([0.0] * 768, EmbeddingZeroVectorError), ([math.nan] * 768, EmbeddingNonFiniteError)],
)
def test_invalid_query_vectors_are_rejected(scope, ctx_a, vector, error):
    with pytest.raises(error):
        retriever(scope, BrokenProvider(query_result=vector)).retrieve("refund", ctx_a, as_of=SEPT)


def test_naive_datetime_as_of_is_rejected(scope, ctx_a):
    with pytest.raises(ValueError):
        retriever(scope, HashingEmbeddingProvider()).retrieve(
            "refund", ctx_a, as_of=datetime(2026, 7, 1)
        )


def test_context_must_be_agent_context(scope, tenant_a):
    with pytest.raises(TypeError):
        retriever(scope, HashingEmbeddingProvider()).retrieve(
            "refund", {"tenant_id": str(tenant_a.tenant_id)}
        )  # type: ignore[arg-type]


def test_retrieval_log_is_safe(scope, ctx_a, caplog):
    with caplog.at_level(logging.INFO, logger="app.knowledge.retrieval"):
        retriever(scope, HashingEmbeddingProvider()).retrieve(
            "PRIVATE-QUERY refund timing", ctx_a, as_of=SEPT
        )
    [rec] = [r for r in caplog.records if r.name == "app.knowledge.retrieval"]
    assert (rec.retriever, rec.result_count, rec.outcome) == ("semantic-pgvector-v1", 5, "ok")
    assert DIGEST_A in rec.profile
    assert "PRIVATE-QUERY" not in caplog.text and "business days" not in caplog.text


@pytest.mark.parametrize("limit", [0, MAX_LIMIT + 1])
def test_vector_query_boundary_enforces_the_same_limit(kb, ctx_a, limit):
    profile = resolve_profile(HashingEmbeddingProvider())
    with pytest.raises(ValueError):
        KnowledgeQueries(kb, ctx_a.tenant).nearest_chunks([1.0] * 768, profile, SEPT, limit)
    assert (
        len(
            KnowledgeQueries(kb, ctx_a.tenant).nearest_chunks([1.0] * 768, profile, SEPT, MAX_LIMIT)
        )
        == MAX_LIMIT
    )
