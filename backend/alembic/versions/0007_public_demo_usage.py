"""Public-demo usage: durable per-visitor message budget for anonymous visitors.

One row per verified anonymous JWT subject (stored only as a SHA-256 digest). Not tenant
data. Downgrade drops the table.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-25 07:43:52.087120

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "public_demo_usage",
        sa.Column("subject_hash", sa.String(length=64), nullable=False),
        sa.Column("messages", sa.Integer(), nullable=False),
        sa.Column(
            "first_used_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_used_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(subject_hash) = 64", name=op.f("ck_public_demo_usage_subject_hash_sha256")
        ),
        sa.CheckConstraint(
            "messages >= 0", name=op.f("ck_public_demo_usage_messages_not_negative")
        ),
        sa.PrimaryKeyConstraint("subject_hash", name=op.f("pk_public_demo_usage")),
    )


def downgrade() -> None:
    op.drop_table("public_demo_usage")
