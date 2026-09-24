"""Tenant-scoped, read-only queries over versioned policy documents and chunks.

Every method is bound to the TenantContext given at construction; there is no generic
document-table query. Temporal selection never reads a clock: callers pass ``as_of``.
"""

import uuid
from datetime import date, datetime

from sqlalchemy import Select, and_, exists, or_, select

from app.core.errors import NotFoundError
from app.knowledge.citations import citation_for, parse_citation
from app.knowledge.embeddings.profile import EmbeddingProfile
from app.knowledge.limits import validate_limit
from app.knowledge.temporal import effective_date
from app.models import KnowledgeChunk, KnowledgeChunkEmbedding, KnowledgeDocument
from app.schemas.knowledge import KnowledgeChunkRead, KnowledgeDocumentRead
from app.services.base import TenantScopedQueries


def effective_filter(as_of: date):  # noqa: ANN201 - SQLAlchemy boolean clause
    """``effective_from <= as_of < effective_to`` (open-ended when effective_to is NULL)."""
    return and_(
        KnowledgeDocument.effective_from <= as_of,
        or_(KnowledgeDocument.effective_to.is_(None), KnowledgeDocument.effective_to > as_of),
    )


class KnowledgeQueries(TenantScopedQueries):
    def _documents(self) -> Select:
        return select(KnowledgeDocument).where(KnowledgeDocument.tenant_id == self._tenant_id)

    def _chunks(self) -> Select:
        return (
            select(KnowledgeChunk, KnowledgeDocument)
            .join(
                KnowledgeDocument,
                and_(
                    KnowledgeDocument.tenant_id == KnowledgeChunk.tenant_id,
                    KnowledgeDocument.id == KnowledgeChunk.document_id,
                ),
            )
            .where(
                KnowledgeChunk.tenant_id == self._tenant_id,
                KnowledgeDocument.tenant_id == self._tenant_id,
            )
        )

    @staticmethod
    def _chunk_read(chunk: KnowledgeChunk, doc: KnowledgeDocument) -> KnowledgeChunkRead:
        return KnowledgeChunkRead(
            chunk_id=chunk.id,
            citation=citation_for(doc.document_key, doc.version, chunk.chunk_index),
            document_key=doc.document_key,
            title=doc.title,
            version=doc.version,
            section=chunk.section,
            chunk_index=chunk.chunk_index,
            effective_from=doc.effective_from,
            effective_to=doc.effective_to,
            content=chunk.content,
        )

    def list_documents(self) -> list[KnowledgeDocumentRead]:
        stmt = self._documents().order_by(KnowledgeDocument.document_key, KnowledgeDocument.version)
        return [KnowledgeDocumentRead.model_validate(d) for d in self._session.scalars(stmt)]

    def get_document(self, document_key: str, version: int) -> KnowledgeDocumentRead:
        doc = self._session.scalar(
            self._documents().where(
                KnowledgeDocument.document_key == document_key,
                KnowledgeDocument.version == version,
            )
        )
        if doc is None:
            raise NotFoundError("knowledge_document", f"{document_key}/v{version}")
        return KnowledgeDocumentRead.model_validate(doc)

    def get_effective_document(
        self, document_key: str, as_of: date | datetime
    ) -> KnowledgeDocumentRead | None:
        """The single version effective on ``as_of`` (None before the first version or in a
        gap). Validated ingestion guarantees ranges never overlap; if they ever did, the
        highest version wins deterministically."""
        day = effective_date(as_of)
        doc = self._session.scalar(
            self._documents()
            .where(KnowledgeDocument.document_key == document_key, effective_filter(day))
            .order_by(KnowledgeDocument.version.desc())
            .limit(1)
        )
        return KnowledgeDocumentRead.model_validate(doc) if doc is not None else None

    def list_chunks(self, document_key: str, version: int) -> list[KnowledgeChunkRead]:
        stmt = (
            self._chunks()
            .where(
                KnowledgeDocument.document_key == document_key,
                KnowledgeDocument.version == version,
            )
            .order_by(KnowledgeChunk.chunk_index)
        )
        return [self._chunk_read(c, d) for c, d in self._session.execute(stmt)]

    def get_chunk(self, chunk_id: uuid.UUID) -> KnowledgeChunkRead:
        row = self._session.execute(self._chunks().where(KnowledgeChunk.id == chunk_id)).first()
        if row is None:
            raise NotFoundError("knowledge_chunk", str(chunk_id))
        return self._chunk_read(*row)

    def get_chunk_by_citation(self, citation: str) -> KnowledgeChunkRead:
        """Resolve a tenant-relative citation inside THIS tenant only."""
        try:
            ref = parse_citation(citation)
        except ValueError:
            raise NotFoundError("knowledge_chunk", "invalid citation") from None
        row = self._session.execute(
            self._chunks().where(
                KnowledgeDocument.document_key == ref.document_key,
                KnowledgeDocument.version == ref.version,
                KnowledgeChunk.chunk_index == ref.chunk_index,
            )
        ).first()
        if row is None:
            raise NotFoundError("knowledge_chunk", citation)
        return self._chunk_read(*row)

    # --- semantic (pgvector) -------------------------------------------------------------
    def _profile_filter(self, profile: EmbeddingProfile):  # noqa: ANN202 - SQL clause
        e = KnowledgeChunkEmbedding
        return and_(
            e.tenant_id == self._tenant_id,
            e.provider == profile.provider,
            e.model == profile.model,
            e.model_digest == profile.model_digest,
            e.dimensions == profile.dimensions,
            e.input_version == profile.input_version,
        )

    def has_embeddings(self, profile: EmbeddingProfile) -> bool:
        """Whether THIS tenant has any vectors in exactly this concrete profile."""
        return bool(self._session.scalar(select(exists().where(self._profile_filter(profile)))))

    def nearest_chunks(
        self,
        query_vector: list[float],
        profile: EmbeddingProfile,
        as_of: date | datetime,
        limit: int,
    ) -> list[tuple[KnowledgeChunkRead, float]]:
        """Exact cosine search (no ANN index), eligibility filtered IN the same statement:
        tenant (all three tables), concrete embedding profile incl. model digest and
        dimensions, and the effective policy version on ``as_of``. Returns (chunk,
        cosine_similarity) with similarity = 1 - cosine distance (``<=>``), in [-1, 1].
        Order: distance ASC, document_key, version DESC, chunk_index (deterministic)."""
        if len(query_vector) != profile.dimensions:
            raise ValueError("query vector dimensions do not match the profile")
        validate_limit(limit)
        e = KnowledgeChunkEmbedding
        distance = e.embedding.cosine_distance(query_vector)
        stmt = (
            self._chunks()
            .add_columns((1 - distance).label("similarity"))
            .join(e, and_(e.tenant_id == KnowledgeChunk.tenant_id, e.chunk_id == KnowledgeChunk.id))
            .where(self._profile_filter(profile), effective_filter(effective_date(as_of)))
            .order_by(
                distance,
                KnowledgeDocument.document_key,
                KnowledgeDocument.version.desc(),
                KnowledgeChunk.chunk_index,
            )
            .limit(limit)
        )
        return [
            (self._chunk_read(chunk, doc), float(similarity))
            for chunk, doc, similarity in self._session.execute(stmt)
        ]
