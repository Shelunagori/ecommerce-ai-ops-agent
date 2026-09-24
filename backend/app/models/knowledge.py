"""Knowledge-domain metadata: versioned policy documents and their derived chunks (Step 7).

No embedding/vector column yet (Step 8). Documents are immutable per (tenant, key, version);
chunks are derived from a document by a recorded chunker configuration (``chunking_hash``).
"""

import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import DocumentType, check_in

SHA256_CHECK = "~ '^[0-9a-f]{64}$'"
KEY_CHECK = "~ '^[a-z0-9]+(-[a-z0-9]+)*$'"


class KnowledgeDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "knowledge_documents"

    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="RESTRICT"))
    document_key: Mapped[str] = mapped_column(String(80))
    title: Mapped[str] = mapped_column(String(200))
    document_type: Mapped[str] = mapped_column(String(16), default=DocumentType.POLICY)
    version: Mapped[int] = mapped_column(Integer)
    # Half-open range: effective_from <= as_of < effective_to (UTC calendar dates).
    effective_from: Mapped[date] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    # Server-side provenance only (relative to the policy directory); never a citation.
    source_name: Mapped[str] = mapped_column(String(255))
    # sha256 of the IMMUTABLE policy payload (tenant, key, title, type, version,
    # effective_from, body) - deliberately NOT effective_to, the one-time retirement field.
    immutable_content_hash: Mapped[str] = mapped_column(String(64))
    chunker: Mapped[str] = mapped_column(String(40))  # e.g. "policy-section-v1"
    chunking_hash: Mapped[str] = mapped_column(String(64))  # chunker version + settings
    chunk_count: Mapped[int] = mapped_column(Integer)

    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),  # target of the tenant-aware chunk FK
        UniqueConstraint("tenant_id", "document_key", "version"),
        CheckConstraint("version > 0", name="version_positive"),
        CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from", name="effective_range"
        ),
        CheckConstraint(f"document_key {KEY_CHECK}", name="document_key_format"),
        CheckConstraint(check_in("document_type", DocumentType), name="document_type_valid"),
        CheckConstraint(
            f"immutable_content_hash {SHA256_CHECK}", name="immutable_content_hash_format"
        ),
        CheckConstraint(f"chunking_hash {SHA256_CHECK}", name="chunking_hash_format"),
        CheckConstraint("chunk_count > 0", name="chunk_count_positive"),
        # Partial defence only: at most ONE open-ended version per (tenant, key). Full
        # range-overlap prevention is enforced by validated ingestion (it would need a
        # btree_gist exclusion constraint, deliberately not added in Step 7).
        Index(
            "uq_knowledge_documents_one_open_version",
            "tenant_id",
            "document_key",
            unique=True,
            postgresql_where=text("effective_to IS NULL"),
        ),
        Index(
            "ix_knowledge_documents_tenant_id_document_key_effective_from",
            "tenant_id",
            "document_key",
            "effective_from",
        ),
    )


class KnowledgeChunk(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "knowledge_chunks"

    tenant_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    chunk_index: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(300))  # heading path, e.g. "Refund Policy > Timing"
    content: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))

    # Chunks are immutable derived rows: created only, never updated.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), sort_order=10
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),  # target of the tenant-aware embedding FK (0003)
        UniqueConstraint("tenant_id", "document_id", "chunk_index"),
        # Tenant-aware FK: a chunk can only reference a document of the SAME tenant.
        ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["knowledge_documents.tenant_id", "knowledge_documents.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("chunk_index >= 0", name="chunk_index_non_negative"),
        CheckConstraint("char_count > 0 AND char_count = char_length(content)", name="char_count"),
        CheckConstraint(f"content_hash {SHA256_CHECK}", name="content_hash_format"),
    )


class KnowledgeChunkEmbedding(UUIDPrimaryKeyMixin, Base):
    """A derived artifact: one chunk's vector in ONE concrete embedding profile.

    Chunks are source-derived knowledge units; embeddings depend on provider, model build
    (digest), dimensions and input format, so they live in their own table. Several
    profiles may coexist for the same chunk; rows are never updated.
    """

    __tablename__ = "knowledge_chunk_embeddings"

    tenant_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    chunk_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))  # configured tag, e.g. "x:latest"
    model_digest: Mapped[str] = mapped_column(String(64))  # resolved immutable build
    dimensions: Mapped[int] = mapped_column(Integer)
    input_version: Mapped[str] = mapped_column(String(40))
    input_hash: Mapped[str] = mapped_column(String(64))  # sha256 of the embedded text
    # Generic (unsized) pgvector column: dimensions are recorded per row and enforced by
    # the vector_dims check, so another profile can use another size later.
    embedding: Mapped[list[float]] = mapped_column(Vector())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), sort_order=10
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "chunk_id",
            "provider",
            "model",
            "model_digest",
            "dimensions",
            "input_version",
            name="uq_knowledge_chunk_embeddings_profile",
        ),
        # Tenant-aware FK: an embedding can only belong to a chunk of the SAME tenant.
        ForeignKeyConstraint(
            ["tenant_id", "chunk_id"],
            ["knowledge_chunks.tenant_id", "knowledge_chunks.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("dimensions BETWEEN 1 AND 16000", name="dimensions_range"),
        CheckConstraint("vector_dims(embedding) = dimensions", name="embedding_dims_match"),
        CheckConstraint("provider ~ '^[a-z0-9-]{1,32}$'", name="provider_format"),
        CheckConstraint("input_version ~ '^[a-z0-9-]{1,40}$'", name="input_version_format"),
        CheckConstraint(f"model_digest {SHA256_CHECK}", name="model_digest_format"),
        CheckConstraint(f"input_hash {SHA256_CHECK}", name="input_hash_format"),
        # Plain btree filter index for profile-scoped search; deliberately NO ANN index
        # (HNSW / IVFFlat) in Step 8: exact search is the evaluation ground truth.
        Index(
            "ix_knowledge_chunk_embeddings_tenant_profile",
            "tenant_id",
            "provider",
            "model",
            "model_digest",
            "dimensions",
            "input_version",
        ),
    )
