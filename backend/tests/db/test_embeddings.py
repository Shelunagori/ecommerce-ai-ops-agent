"""pgvector storage and embedding materialization on real PostgreSQL (fake providers only).

Every test runs in a rolled-back transaction with the real synthetic corpus ingested."""

import logging
import math
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from app.knowledge.chunking import ChunkingConfig
from app.knowledge.embeddings.errors import (
    EmbeddingCountMismatchError,
    EmbeddingDimensionMismatchError,
    EmbeddingNonFiniteError,
    EmbeddingStaleConflictError,
    EmbeddingTimeoutError,
    EmbeddingZeroVectorError,
)
from app.knowledge.embeddings.inputs import document_input, input_hash
from app.knowledge.embeddings.materialize import embedding_id_for, materialize_embeddings
from app.knowledge.embeddings.provider import resolve_profile
from app.knowledge.ingest import ingest_policies
from app.knowledge.sources import DEFAULT_POLICY_DIR
from app.models import KnowledgeChunk, KnowledgeChunkEmbedding, KnowledgeDocument
from tests.knowledge.fakes import DIGEST_A, DIGEST_B, BrokenProvider, HashingEmbeddingProvider

E = KnowledgeChunkEmbedding


@pytest.fixture
def kb(db_session):
    ingest_policies(db_session, DEFAULT_POLICY_DIR, ChunkingConfig())
    return db_session


def count(session, **filters) -> int:
    stmt = select(func.count()).select_from(E)
    for k, v in filters.items():
        stmt = stmt.where(getattr(E, k) == v)
    return session.scalar(stmt)


def chunk_of(session, tenant_id, key="refund-policy", version=1, index=1) -> KnowledgeChunk:
    return session.scalar(
        select(KnowledgeChunk)
        .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
        .where(
            KnowledgeChunk.tenant_id == tenant_id,
            KnowledgeDocument.document_key == key,
            KnowledgeDocument.version == version,
            KnowledgeChunk.chunk_index == index,
        )
    )


def row(chunk: KnowledgeChunk, digest=DIGEST_A, dims=768, vector=None, **kw) -> E:
    values = {
        "id": uuid.uuid4(),
        "tenant_id": chunk.tenant_id,
        "chunk_id": chunk.id,
        "provider": "ollama",
        "model": "nomic-embed-text-v2-moe:latest",
        "model_digest": digest,
        "dimensions": dims,
        "input_version": "policy-embedding-input-v1",
        "input_hash": "c" * 64,
        "embedding": vector or [1.0] + [0.0] * (dims - 1),
    }
    values.update(kw)
    return E(**values)


# --- extension and schema --------------------------------------------------------------------
def test_vector_extension_is_enabled(kb):
    assert kb.scalar(text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")) == 1
    col_type = kb.scalar(
        text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'knowledge_chunk_embeddings'::regclass AND attname = 'embedding'"
        )
    )
    assert col_type == "vector"  # generic, not vector(768)


def test_no_ann_index_exists(kb):
    defs = kb.scalars(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'knowledge_chunk_embeddings'")
    ).all()
    assert defs and not any(("hnsw" in d or "ivfflat" in d) for d in defs)


# --- materialization ------------------------------------------------------------------------
def test_materialization_embeds_every_chunk_once(kb):
    fake = HashingEmbeddingProvider()
    report = materialize_embeddings(kb, fake, batch_size=16)
    assert (report.total_chunks, report.embedded, report.unchanged, report.batches) == (
        51,
        51,
        0,
        4,
    )
    assert [len(c) for c in fake.document_calls] == [16, 16, 16, 3]
    assert fake.digest_calls == 1 and count(kb) == 51
    rows = kb.scalars(select(E)).all()
    profile = resolve_profile(fake)
    for r in rows:
        assert r.id == embedding_id_for(r.chunk_id, profile)
        assert (r.provider, r.model, r.model_digest, r.dimensions, r.input_version) == (
            "ollama",
            "nomic-embed-text-v2-moe:latest",
            DIGEST_A,
            768,
            "policy-embedding-input-v1",
        )
        assert len(r.embedding) == 768


def test_rerun_is_idempotent_and_makes_no_embedding_calls(kb):
    materialize_embeddings(kb, HashingEmbeddingProvider(), batch_size=16)
    before = sorted((r.id, r.input_hash, r.created_at) for r in kb.scalars(select(E)))
    fake = HashingEmbeddingProvider()
    report = materialize_embeddings(kb, fake, batch_size=16)
    assert (report.embedded, report.unchanged, report.batches) == (0, 51, 0)
    assert fake.document_calls == []
    assert sorted((r.id, r.input_hash, r.created_at) for r in kb.scalars(select(E))) == before


def test_inputs_are_prefixed_with_context_and_carry_no_identifiers(kb, tenant_a):
    fake = HashingEmbeddingProvider()
    materialize_embeddings(kb, fake, batch_size=64)
    inputs = [t for batch in fake.document_calls for t in batch]
    chunk = chunk_of(kb, tenant_a.tenant_id)
    doc = kb.get(KnowledgeDocument, chunk.document_id)
    expected = document_input(doc.title, chunk.section, chunk.content)
    assert expected in inputs
    assert expected.startswith("search_document: Title: Refund Policy\nSection: Refund Policy > ")
    stored = kb.scalar(select(E.input_hash).where(E.chunk_id == chunk.id))
    assert stored == input_hash(expected)
    blob = "\n".join(inputs)
    for leak in (str(tenant_a.tenant_id), str(chunk.id), "policy://", ".md", "northstar-commerce"):
        assert leak not in blob


def test_vectors_round_trip_and_dimension_metadata_matches(kb):
    materialize_embeddings(kb, HashingEmbeddingProvider(), batch_size=16)
    mismatched = kb.scalar(
        text(
            "SELECT count(*) FROM knowledge_chunk_embeddings "
            "WHERE vector_dims(embedding) <> dimensions"
        )
    )
    assert mismatched == 0
    r = kb.scalars(select(E)).first()
    kb.expire_all()
    again = kb.get(E, r.id)
    assert again.embedding == pytest.approx(r.embedding, rel=1e-6)


def test_a_new_digest_is_a_new_profile_and_old_vectors_stay(kb):
    materialize_embeddings(kb, HashingEmbeddingProvider(digest=DIGEST_A), batch_size=16)
    old = sorted((r.id, tuple(r.embedding)) for r in kb.scalars(select(E)))
    report = materialize_embeddings(kb, HashingEmbeddingProvider(digest=DIGEST_B), batch_size=16)
    assert report.embedded == 51
    assert count(kb, model_digest=DIGEST_A) == 51 and count(kb, model_digest=DIGEST_B) == 51
    assert (
        sorted(
            (r.id, tuple(r.embedding))
            for r in kb.scalars(select(E).where(E.model_digest == DIGEST_A))
        )
        == old
    )


def test_other_dimensions_or_models_coexist(kb):
    materialize_embeddings(kb, HashingEmbeddingProvider(), batch_size=16)
    materialize_embeddings(kb, HashingEmbeddingProvider(dimensions=256), batch_size=16)
    materialize_embeddings(kb, HashingEmbeddingProvider(model_name="other-embed:v1"), batch_size=16)
    assert count(kb) == 153
    assert count(kb, dimensions=256) == 51 and count(kb, model="other-embed:v1") == 51


def test_stale_input_is_a_conflict_and_nothing_is_overwritten(kb):
    materialize_embeddings(kb, HashingEmbeddingProvider(), batch_size=16)
    victim = kb.scalars(select(E)).first()
    kb.execute(
        text("UPDATE knowledge_chunk_embeddings SET input_hash = :h WHERE id = :id"),
        {"h": "d" * 64, "id": victim.id},
    )
    kb.expire_all()
    before = sorted((r.id, r.input_hash) for r in kb.scalars(select(E)))
    fake = HashingEmbeddingProvider()
    with pytest.raises(EmbeddingStaleConflictError) as exc:
        materialize_embeddings(kb, fake, batch_size=16)
    assert exc.value.code == "embedding_stale_conflict" and exc.value.report.conflicts == [
        str(victim.chunk_id)
    ]
    assert fake.document_calls == []  # refused before any model call
    assert sorted((r.id, r.input_hash) for r in kb.scalars(select(E))) == before


def test_stale_conflict_in_another_profile_does_not_block(kb):
    materialize_embeddings(kb, HashingEmbeddingProvider(digest=DIGEST_B), batch_size=16)
    kb.execute(text("UPDATE knowledge_chunk_embeddings SET input_hash = :h"), {"h": "d" * 64})
    report = materialize_embeddings(kb, HashingEmbeddingProvider(digest=DIGEST_A), batch_size=16)
    assert report.embedded == 51


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        (lambda texts: [[1.0] * 768 for _ in texts[:-1]], EmbeddingCountMismatchError),
        (lambda texts: [[1.0] * 767 for _ in texts], EmbeddingDimensionMismatchError),
        (lambda texts: [[math.nan] * 768 for _ in texts], EmbeddingNonFiniteError),
        (lambda texts: [[math.inf] + [0.0] * 767 for _ in texts], EmbeddingNonFiniteError),
        (lambda texts: [[0.0] * 768 for _ in texts], EmbeddingZeroVectorError),
    ],
)
def test_invalid_vectors_abort_the_whole_run(kb, bad, error):
    with pytest.raises(error):
        materialize_embeddings(kb, BrokenProvider(documents_result=bad), batch_size=16)
    kb.flush()
    assert count(kb) == 0


def test_a_failing_later_batch_writes_nothing(kb):
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 3:
            raise EmbeddingTimeoutError()
        return [[1.0] + [0.0] * 767 for _ in texts]

    with pytest.raises(EmbeddingTimeoutError):
        materialize_embeddings(kb, BrokenProvider(documents_result=flaky), batch_size=16)
    assert count(kb) == 0 and calls["n"] == 3


def test_batch_size_bounds(kb):
    for bad in (0, 65):
        with pytest.raises(ValueError):
            materialize_embeddings(kb, HashingEmbeddingProvider(), batch_size=bad)


def test_materialization_log_is_safe(kb, caplog):
    with caplog.at_level(logging.INFO, logger="app.knowledge.embeddings"):
        materialize_embeddings(kb, HashingEmbeddingProvider(), batch_size=16)
    [rec] = [r for r in caplog.records if r.msg == "embedding materialization"]
    assert (rec.outcome, rec.embedded, rec.batches, rec.batch_size) == ("ok", 51, 4, 16)
    assert DIGEST_A in rec.profile
    assert "business days" not in caplog.text and "0.0, 0.0" not in caplog.text


# --- database constraints ----------------------------------------------------------------------
def test_same_profile_cannot_be_duplicated_for_a_chunk(kb, tenant_a):
    chunk = chunk_of(kb, tenant_a.tenant_id)
    kb.add(row(chunk))
    kb.flush()
    kb.add(row(chunk))
    with pytest.raises(IntegrityError):
        kb.flush()


def test_same_chunk_may_have_several_profiles(kb, tenant_a):
    chunk = chunk_of(kb, tenant_a.tenant_id)
    kb.add_all(
        [
            row(chunk),
            row(chunk, digest=DIGEST_B),
            row(chunk, dims=3, vector=[1, 2, 3]),
            row(chunk, input_version="policy-embedding-input-v2"),
        ]
    )
    kb.flush()
    assert count(kb, chunk_id=chunk.id) == 4


def test_embedding_cannot_attach_to_another_tenants_chunk(kb, tenant_a, tenant_b):
    b_chunk = chunk_of(kb, tenant_b.tenant_id)
    kb.add(row(b_chunk, tenant_id=tenant_a.tenant_id))
    with pytest.raises(IntegrityError):
        kb.flush()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dims": 768, "vector": [1.0, 0.0, 0.0]},  # vector size != recorded dimensions
        {"dims": 0, "vector": [1.0]},
        {"model_digest": "xyz"},
        {"input_hash": "short"},
        {"provider": "Bad Provider"},
    ],
)
def test_embedding_check_constraints(kb, tenant_a, kwargs):
    chunk = chunk_of(kb, tenant_a.tenant_id)
    dims = kwargs.pop("dims", 768)
    vector = kwargs.pop("vector", None)
    kb.add(row(chunk, dims=dims, vector=vector, **kwargs))
    with pytest.raises(IntegrityError):
        kb.flush()


def test_database_rejects_non_finite_vectors(kb, tenant_a):
    chunk = chunk_of(kb, tenant_a.tenant_id)
    kb.add(row(chunk, dims=3, vector=[1.0, math.nan, 0.0]))
    with pytest.raises(Exception):  # noqa: B017,PT011 - driver/pgvector error type varies
        kb.flush()


def test_created_at_is_set(kb, tenant_a):
    chunk = chunk_of(kb, tenant_a.tenant_id)
    r = row(chunk)
    kb.add(r)
    kb.flush()
    kb.refresh(r)
    assert r.created_at <= datetime.now(UTC)
