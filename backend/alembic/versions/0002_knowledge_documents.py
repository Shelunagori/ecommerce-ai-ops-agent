"""Knowledge domain: versioned policy documents and their derived chunks (Step 7).

Metadata only: no extensions (pgvector is NOT enabled here) and no embedding column.
Tenant integrity: chunks reference documents through the composite (tenant_id, id) FK.
Range overlap between versions is validated by ingestion; the database adds a partial
unique index (one open-ended version per tenant/document_key) as a defence in depth.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-24 05:05:24.490361

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("document_key", sa.String(length=80), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("document_type", sa.String(length=16), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("source_name", sa.String(length=255), nullable=False),
        sa.Column("immutable_content_hash", sa.String(length=64), nullable=False),
        sa.Column("chunker", sa.String(length=40), nullable=False),
        sa.Column("chunking_hash", sa.String(length=64), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "chunking_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_knowledge_documents_chunking_hash_format"),
        ),
        sa.CheckConstraint(
            "immutable_content_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_knowledge_documents_immutable_content_hash_format"),
        ),
        sa.CheckConstraint(
            "document_key ~ '^[a-z0-9]+(-[a-z0-9]+)*$'",
            name=op.f("ck_knowledge_documents_document_key_format"),
        ),
        sa.CheckConstraint(
            "document_type IN ('policy')", name=op.f("ck_knowledge_documents_document_type_valid")
        ),
        sa.CheckConstraint(
            "chunk_count > 0", name=op.f("ck_knowledge_documents_chunk_count_positive")
        ),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from",
            name=op.f("ck_knowledge_documents_effective_range"),
        ),
        sa.CheckConstraint("version > 0", name=op.f("ck_knowledge_documents_version_positive")),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_knowledge_documents_tenant_id_tenants"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_documents")),
        sa.UniqueConstraint(
            "tenant_id",
            "document_key",
            "version",
            name=op.f("uq_knowledge_documents_tenant_id_document_key_version"),
        ),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_knowledge_documents_tenant_id_id")),
    )
    op.create_index(
        "ix_knowledge_documents_tenant_id_document_key_effective_from",
        "knowledge_documents",
        ["tenant_id", "document_key", "effective_from"],
        unique=False,
    )
    op.create_index(
        "uq_knowledge_documents_one_open_version",
        "knowledge_documents",
        ["tenant_id", "document_key"],
        unique=True,
        postgresql_where=sa.text("effective_to IS NULL"),
    )
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(length=300), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'", name=op.f("ck_knowledge_chunks_content_hash_format")
        ),
        sa.CheckConstraint(
            "char_count > 0 AND char_count = char_length(content)",
            name=op.f("ck_knowledge_chunks_char_count"),
        ),
        sa.CheckConstraint(
            "chunk_index >= 0", name=op.f("ck_knowledge_chunks_chunk_index_non_negative")
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["knowledge_documents.tenant_id", "knowledge_documents.id"],
            name=op.f("fk_knowledge_chunks_tenant_id_document_id_knowledge_documents"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_chunks")),
        sa.UniqueConstraint(
            "tenant_id",
            "document_id",
            "chunk_index",
            name=op.f("uq_knowledge_chunks_tenant_id_document_id_chunk_index"),
        ),
    )


def downgrade() -> None:
    op.drop_table("knowledge_chunks")
    op.drop_index(
        "uq_knowledge_documents_one_open_version",
        table_name="knowledge_documents",
        postgresql_where=sa.text("effective_to IS NULL"),
    )
    op.drop_index(
        "ix_knowledge_documents_tenant_id_document_key_effective_from",
        table_name="knowledge_documents",
    )
    op.drop_table("knowledge_documents")
