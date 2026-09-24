"""Atomic, idempotent ingestion of validated policy sources into knowledge tables.

Per source (tenant, document_key, version):
* not stored                                  -> ``inserted`` (document + chunks)
* stored, same immutable_content_hash, same chunking, same effective_to
                                              -> ``unchanged`` (nothing written)
* stored open-ended, file now sets effective_to (same immutable_content_hash)
                                              -> ``retired``: the ONLY permitted change to a
  stored version (its end date is set once, when a successor is published)
* stored, different immutable_content_hash (body, title, effective_from, ...), or an
  already-set end date that changes or disappears
                                              -> ``source_conflict``: an ingested version
  is immutable; publish a new version instead
* stored, same content, different chunking    -> ``chunking_mismatch``: the SOURCE did not
  change, but the stored chunks were materialised by another chunker/config. Automatic
  ingestion never replaces them (a future explicit re-chunk operation would).

Everything is validated (sources, version ranges including stored versions, conflicts)
before anything is written. Any conflict aborts the whole run with no changes. Ingestion
never deletes or updates rows; the caller owns the transaction (``unit_of_work``).
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.chunking import CHUNKER, ChunkingConfig, PolicyChunk, chunk_policy
from app.knowledge.sources import PolicySource, load_policy_sources, validate_version_ranges
from app.models import KnowledgeChunk, KnowledgeDocument, Tenant

logger = logging.getLogger("app.knowledge.ingest")

# Fixed namespace: changing it would change every knowledge id.
KNOWLEDGE_NAMESPACE = uuid.UUID("2c4b8a7e-5d1f-4e3a-9b6c-7a0de1c0ffee")

Outcome = Literal["inserted", "unchanged", "retired", "source_conflict", "chunking_mismatch"]


def document_id_for(tenant_id: uuid.UUID, document_key: str, version: int) -> uuid.UUID:
    return uuid.uuid5(KNOWLEDGE_NAMESPACE, f"document/{tenant_id}/{document_key}/v{version}")


def chunk_id_for(document_id: uuid.UUID, index: int, chunk_content_hash: str) -> uuid.UUID:
    return uuid.uuid5(KNOWLEDGE_NAMESPACE, f"chunk/{document_id}/{index}/{chunk_content_hash}")


@dataclass(frozen=True)
class DocumentOutcome:
    tenant: str
    document_key: str
    version: int
    outcome: Outcome
    chunk_count: int


@dataclass
class IngestionReport:
    chunker: str
    chunking_hash: str
    documents: list[DocumentOutcome] = field(default_factory=list)

    @property
    def conflicts(self) -> list[DocumentOutcome]:
        return [d for d in self.documents if d.outcome in ("source_conflict", "chunking_mismatch")]

    def count(self, outcome: Outcome) -> int:
        return sum(d.outcome == outcome for d in self.documents)


class IngestionConflictError(Exception):
    """Raised before any write when at least one source conflicts with stored knowledge."""

    def __init__(self, report: IngestionReport) -> None:
        self.report = report
        names = ", ".join(
            f"{d.tenant}/{d.document_key}/v{d.version}:{d.outcome}" for d in report.conflicts
        )
        super().__init__(f"ingestion refused: {names}")


@dataclass(frozen=True)
class _Stored:
    tenant: str
    document_key: str
    version: int
    effective_from: date
    effective_to: date | None


def ingest_policies(
    session: Session, root: Path, config: ChunkingConfig | None = None
) -> IngestionReport:
    config = config or ChunkingConfig.from_settings()
    tenants = {slug: tid for tid, slug in session.execute(select(Tenant.id, Tenant.slug))}
    slugs = {tid: slug for slug, tid in tenants.items()}
    sources = load_policy_sources(root, tenants)

    stored = {
        (d.tenant_id, d.document_key, d.version): d
        for d in session.scalars(select(KnowledgeDocument))
    }
    source_keys = {
        (tenants[s.metadata.tenant], s.metadata.document_key, s.metadata.version) for s in sources
    }
    # Version ranges are validated across the files AND stored versions without a file.
    validate_version_ranges(
        [(s.metadata, s.source_name) for s in sources]
        + [
            (
                _Stored(
                    slugs[d.tenant_id], d.document_key, d.version, d.effective_from, d.effective_to
                ),
                f"stored:{slugs[d.tenant_id]}/{d.document_key}/v{d.version}",
            )
            for key, d in stored.items()
            if key not in source_keys
        ]
    )

    report = IngestionReport(CHUNKER, config.chunking_hash)
    to_insert: list[tuple[PolicySource, uuid.UUID, list[PolicyChunk]]] = []
    to_retire: list[tuple[KnowledgeDocument, date]] = []
    for source in sources:
        m = source.metadata
        tenant_id = tenants[m.tenant]
        chunks = chunk_policy(source.body, m.title, config)
        existing = stored.get((tenant_id, m.document_key, m.version))
        if existing is None:
            outcome: Outcome = "inserted"
            to_insert.append((source, tenant_id, chunks))
        elif existing.immutable_content_hash != source.immutable_content_hash:
            outcome = "source_conflict"
        elif existing.chunking_hash != config.chunking_hash or existing.chunker != CHUNKER:
            outcome = "chunking_mismatch"
        elif existing.effective_to == m.effective_to:
            outcome = "unchanged"
        elif existing.effective_to is None:
            outcome = "retired"
            to_retire.append((existing, m.effective_to))
        else:
            outcome = "source_conflict"  # a set end date may never change or disappear
        count = existing.chunk_count if existing is not None else len(chunks)
        report.documents.append(
            DocumentOutcome(m.tenant, m.document_key, m.version, outcome, count)
        )
        _log(tenant_id, m.document_key, m.version, outcome, count)

    if report.conflicts:
        raise IngestionConflictError(report)

    for document, effective_to in to_retire:  # before inserts: frees the open-version slot
        document.effective_to = effective_to
    session.flush()
    for source, tenant_id, chunks in to_insert:
        _insert(session, source, tenant_id, chunks, config)
    session.flush()
    return report


def _insert(
    session: Session,
    source: PolicySource,
    tenant_id: uuid.UUID,
    chunks: list[PolicyChunk],
    config: ChunkingConfig,
) -> None:
    m = source.metadata
    doc_id = document_id_for(tenant_id, m.document_key, m.version)
    session.add(
        KnowledgeDocument(
            id=doc_id,
            tenant_id=tenant_id,
            document_key=m.document_key,
            title=m.title,
            document_type=m.document_type,
            version=m.version,
            effective_from=m.effective_from,
            effective_to=m.effective_to,
            source_name=source.source_name,
            immutable_content_hash=source.immutable_content_hash,
            chunker=CHUNKER,
            chunking_hash=config.chunking_hash,
            chunk_count=len(chunks),
        )
    )
    session.flush()  # parent row before the tenant-aware composite FK of its chunks
    session.add_all(
        KnowledgeChunk(
            id=chunk_id_for(doc_id, c.index, c.content_hash),
            tenant_id=tenant_id,
            document_id=doc_id,
            chunk_index=c.index,
            section=c.section,
            content=c.content,
            char_count=c.char_count,
            content_hash=c.content_hash,
        )
        for c in chunks
    )


def _log(tenant_id: uuid.UUID, key: str, version: int, outcome: str, chunk_count: int) -> None:
    logger.log(
        logging.INFO if outcome in ("inserted", "unchanged", "retired") else logging.WARNING,
        "knowledge document",
        extra={
            "tenant_id": str(tenant_id),
            "document_key": key,
            "version": version,
            "outcome": outcome,
            "chunk_count": chunk_count,
        },
    )
