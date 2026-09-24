"""Knowledge chunk embeddings on pgvector (Step 8).

* Enables the PostgreSQL ``vector`` extension (``CREATE EXTENSION IF NOT EXISTS``).
* Adds ``knowledge_chunk_embeddings``: derived vectors per chunk and concrete embedding
  profile (provider, model tag, resolved model digest, dimensions, input version), with a
  generic (unsized) ``vector`` column whose size must equal the row's ``dimensions``.
* Adds ``UNIQUE (tenant_id, id)`` on ``knowledge_chunks`` as the target of the tenant-aware
  composite FK ``(tenant_id, chunk_id)``.
* No ANN index (HNSW / IVFFlat): exact cosine search is the Step-8 evaluation baseline.

Asymmetric downgrade (intentional): downgrade drops the table, its index and the added
unique constraint, but NOT the ``vector`` extension. ``IF NOT EXISTS`` means this migration
cannot prove it created the extension (it may predate it or be shared with another schema
or application, e.g. managed PostgreSQL where it is pre-installed), and dropping a shared
extension would destroy other objects. Removing it is a deliberate manual DBA action.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-24 06:32:21.230229

"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_unique_constraint(
        op.f("uq_knowledge_chunks_tenant_id_id"), "knowledge_chunks", ["tenant_id", "id"]
    )
    op.create_table(
        "knowledge_chunk_embeddings",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("model_digest", sa.String(length=64), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("input_version", sa.String(length=40), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "input_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_knowledge_chunk_embeddings_input_hash_format"),
        ),
        sa.CheckConstraint(
            "input_version ~ '^[a-z0-9-]{1,40}$'",
            name=op.f("ck_knowledge_chunk_embeddings_input_version_format"),
        ),
        sa.CheckConstraint(
            "model_digest ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_knowledge_chunk_embeddings_model_digest_format"),
        ),
        sa.CheckConstraint(
            "provider ~ '^[a-z0-9-]{1,32}$'",
            name=op.f("ck_knowledge_chunk_embeddings_provider_format"),
        ),
        sa.CheckConstraint(
            "dimensions BETWEEN 1 AND 16000",
            name=op.f("ck_knowledge_chunk_embeddings_dimensions_range"),
        ),
        sa.CheckConstraint(
            "vector_dims(embedding) = dimensions",
            name=op.f("ck_knowledge_chunk_embeddings_embedding_dims_match"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "chunk_id"],
            ["knowledge_chunks.tenant_id", "knowledge_chunks.id"],
            name=op.f("fk_knowledge_chunk_embeddings_tenant_id_chunk_id_knowledge_chunks"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_chunk_embeddings")),
        sa.UniqueConstraint(
            "tenant_id",
            "chunk_id",
            "provider",
            "model",
            "model_digest",
            "dimensions",
            "input_version",
            name="uq_knowledge_chunk_embeddings_profile",
        ),
    )
    op.create_index(
        "ix_knowledge_chunk_embeddings_tenant_profile",
        "knowledge_chunk_embeddings",
        ["tenant_id", "provider", "model", "model_digest", "dimensions", "input_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_knowledge_chunk_embeddings_tenant_profile", table_name="knowledge_chunk_embeddings"
    )
    op.drop_table("knowledge_chunk_embeddings")
    op.drop_constraint(op.f("uq_knowledge_chunks_tenant_id_id"), "knowledge_chunks", type_="unique")
    # The vector extension is intentionally kept (see module docstring).
