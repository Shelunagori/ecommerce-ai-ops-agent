"""Tenant memberships (Phase 7): verified user subject -> allowed tenant + role.

The trusted tenant boundary for authenticated (Supabase JWT) mode: a tenant is usable by a
user only through a row here. Downgrade drops the table.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-24 10:56:03.280917

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tenant_memberships",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("user_subject", sa.String(length=128), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role IN ('member', 'approver')", name=op.f("ck_tenant_memberships_role_valid")
        ),
        sa.CheckConstraint(
            "length(user_subject) BETWEEN 1 AND 128",
            name=op.f("ck_tenant_memberships_subject_not_empty"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_tenant_memberships_tenant_id_tenants"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tenant_memberships")),
        sa.UniqueConstraint(
            "tenant_id", "user_subject", name=op.f("uq_tenant_memberships_tenant_id_user_subject")
        ),
    )
    op.create_index(
        "ix_tenant_memberships_user_subject", "tenant_memberships", ["user_subject"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_tenant_memberships_user_subject", table_name="tenant_memberships")
    op.drop_table("tenant_memberships")
