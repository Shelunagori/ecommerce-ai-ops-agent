"""Deterministic, atomic, idempotent materialization of chunk embeddings for one profile.

1. Resolve the provider's CONCRETE profile (model digest resolved once, here).
2. Read every ingested chunk (all tenants) with its document title, in a stable order,
   and build its ``policy-embedding-input-v1`` text and ``input_hash``.
3. Classify each chunk against stored rows of THIS profile:
   missing -> embed; same input_hash -> unchanged; different input_hash -> stale conflict.
   Any conflict aborts before a single provider call (never silently overwritten).
4. Embed the missing inputs in batches; validate count, dimensions, finiteness and
   non-zero norm for every vector.
5. Insert all new rows (caller owns the single transaction). Rows of other profiles
   (other models, digests, dimensions or input versions) are never touched or deleted.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.knowledge.embeddings.errors import EmbeddingStaleConflictError
from app.knowledge.embeddings.inputs import document_input, input_hash
from app.knowledge.embeddings.profile import EmbeddingProfile
from app.knowledge.embeddings.provider import (
    EmbeddingProvider,
    embed_document_inputs,
    resolve_profile,
)
from app.knowledge.ingest import KNOWLEDGE_NAMESPACE
from app.models import KnowledgeChunk, KnowledgeChunkEmbedding, KnowledgeDocument

logger = logging.getLogger("app.knowledge.embeddings")


def embedding_id_for(chunk_id: uuid.UUID, profile: EmbeddingProfile) -> uuid.UUID:
    return uuid.uuid5(KNOWLEDGE_NAMESPACE, f"embedding/{chunk_id}/{profile.key}")


@dataclass(frozen=True)
class _Pending:
    tenant_id: uuid.UUID
    chunk_id: uuid.UUID
    text: str
    input_hash: str


@dataclass
class MaterializationReport:
    profile: EmbeddingProfile
    total_chunks: int = 0
    embedded: int = 0
    unchanged: int = 0
    conflicts: list[str] = field(default_factory=list)  # chunk ids (never content)
    batches: int = 0
    batch_size: int = 0
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile.as_dict(),
            "total_chunks": self.total_chunks,
            "embedded": self.embedded,
            "unchanged": self.unchanged,
            "conflicts": len(self.conflicts),
            "dimensions": self.profile.dimensions,
            "batches": self.batches,
            "batch_size": self.batch_size,
            "duration_ms": self.duration_ms,
        }


def materialize_embeddings(
    session: Session, provider: EmbeddingProvider, *, batch_size: int = 16
) -> MaterializationReport:
    if not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be between 1 and 64")
    started = time.perf_counter()
    profile = resolve_profile(provider)
    report = MaterializationReport(profile, batch_size=batch_size)

    rows = session.execute(
        select(
            KnowledgeChunk.tenant_id,
            KnowledgeChunk.id,
            KnowledgeChunk.section,
            KnowledgeChunk.content,
            KnowledgeDocument.title,
        )
        .join(
            KnowledgeDocument,
            and_(
                KnowledgeDocument.tenant_id == KnowledgeChunk.tenant_id,
                KnowledgeDocument.id == KnowledgeChunk.document_id,
            ),
        )
        .order_by(
            KnowledgeChunk.tenant_id,
            KnowledgeDocument.document_key,
            KnowledgeDocument.version,
            KnowledgeChunk.chunk_index,
        )
    ).all()
    e = KnowledgeChunkEmbedding
    stored = dict(
        session.execute(
            select(e.chunk_id, e.input_hash).where(
                e.provider == profile.provider,
                e.model == profile.model,
                e.model_digest == profile.model_digest,
                e.dimensions == profile.dimensions,
                e.input_version == profile.input_version,
            )
        ).all()
    )

    pending: list[_Pending] = []
    for tenant_id, chunk_id, section, content, title in rows:
        text = document_input(title, section, content)
        digest = input_hash(text)
        existing = stored.get(chunk_id)
        if existing is None:
            pending.append(_Pending(tenant_id, chunk_id, text, digest))
        elif existing == digest:
            report.unchanged += 1
        else:
            report.conflicts.append(str(chunk_id))
    report.total_chunks = len(rows)

    if report.conflicts:
        report.duration_ms = _ms(started)
        _log(report, "stale_conflict")
        raise EmbeddingStaleConflictError(report)

    vectors: list[list[float]] = []
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        vectors.extend(embed_document_inputs(provider, [p.text for p in batch]))
        report.batches += 1

    session.add_all(
        KnowledgeChunkEmbedding(
            id=embedding_id_for(p.chunk_id, profile),
            tenant_id=p.tenant_id,
            chunk_id=p.chunk_id,
            provider=profile.provider,
            model=profile.model,
            model_digest=profile.model_digest,
            dimensions=profile.dimensions,
            input_version=profile.input_version,
            input_hash=p.input_hash,
            embedding=vector,
        )
        for p, vector in zip(pending, vectors, strict=True)
    )
    session.flush()
    report.embedded = len(pending)
    report.duration_ms = _ms(started)
    _log(report, "ok")
    return report


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _log(report: MaterializationReport, outcome: str) -> None:
    # Profile, counts and timing only: never vectors or policy content.
    logger.log(
        logging.INFO if outcome == "ok" else logging.WARNING,
        "embedding materialization",
        extra={
            "profile": report.profile.key,
            "total_chunks": report.total_chunks,
            "embedded": report.embedded,
            "unchanged": report.unchanged,
            "conflicts": len(report.conflicts),
            "batches": report.batches,
            "batch_size": report.batch_size,
            "outcome": outcome,
            "duration_ms": report.duration_ms,
        },
    )
