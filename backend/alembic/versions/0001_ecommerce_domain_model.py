"""Ecommerce domain model: tenants, customers, products, orders, items, invoices, shipments.

Standard PostgreSQL only (gen_random_uuid() is core since PG 13); no extensions.
Tenant integrity: child tables reference parents through composite (tenant_id, id)
foreign keys, so a row can never point at a row owned by another tenant.

Revision ID: 0001
Revises:
Create Date: 2026-09-23 12:18:30.722295

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("slug", sa.String(length=63), nullable=False),
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
            "slug ~ '^[a-z0-9]+(-[a-z0-9]+)*$'", name=op.f("ck_tenants_slug_format")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tenants")),
        sa.UniqueConstraint("slug", name=op.f("uq_tenants_slug")),
    )
    op.create_table(
        "customers",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("customer_code", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
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
            "status IN ('active', 'inactive')", name=op.f("ck_customers_status_valid")
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_customers_tenant_id_tenants"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_customers")),
        sa.UniqueConstraint(
            "tenant_id", "customer_code", name=op.f("uq_customers_tenant_id_customer_code")
        ),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_customers_tenant_id_id")),
    )
    op.create_index(
        "ix_customers_tenant_id_email", "customers", ["tenant_id", "email"], unique=False
    )
    op.create_index("ix_customers_tenant_id_name", "customers", ["tenant_id", "name"], unique=False)
    op.create_table(
        "products",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("sku", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("unit_price", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
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
        sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name=op.f("ck_products_currency_iso4217")),
        sa.CheckConstraint("unit_price >= 0", name=op.f("ck_products_unit_price_non_negative")),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name=op.f("fk_products_tenant_id_tenants"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_products")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_products_tenant_id_id")),
        sa.UniqueConstraint("tenant_id", "sku", name=op.f("uq_products_tenant_id_sku")),
    )
    op.create_index("ix_products_tenant_id_name", "products", ["tenant_id", "name"], unique=False)
    op.create_table(
        "orders",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("customer_id", sa.Uuid(), nullable=False),
        sa.Column("order_number", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("total_amount", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("placed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name=op.f("ck_orders_currency_iso4217")),
        sa.CheckConstraint(
            "status = 'draft' OR placed_at IS NOT NULL",
            name=op.f("ck_orders_placed_at_unless_draft"),
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'confirmed', 'processing', 'shipped', 'delivered', 'cancelled')",
            name=op.f("ck_orders_status_valid"),
        ),
        sa.CheckConstraint("total_amount >= 0", name=op.f("ck_orders_total_amount_non_negative")),
        sa.ForeignKeyConstraint(
            ["tenant_id", "customer_id"],
            ["customers.tenant_id", "customers.id"],
            name=op.f("fk_orders_tenant_id_customer_id_customers"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_orders")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_orders_tenant_id_id")),
        sa.UniqueConstraint(
            "tenant_id", "order_number", name=op.f("uq_orders_tenant_id_order_number")
        ),
    )
    op.create_index(
        "ix_orders_tenant_id_customer_id_placed_at",
        "orders",
        ["tenant_id", "customer_id", "placed_at"],
        unique=False,
    )
    op.create_index(
        "ix_orders_tenant_id_placed_at", "orders", ["tenant_id", "placed_at"], unique=False
    )
    op.create_index("ix_orders_tenant_id_status", "orders", ["tenant_id", "status"], unique=False)
    op.create_table(
        "invoices",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("order_id", sa.Uuid(), nullable=False),
        sa.Column("invoice_number", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("amount", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True),
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
            "(status = 'paid') = (paid_at IS NOT NULL)", name=op.f("ck_invoices_paid_at_iff_paid")
        ),
        sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name=op.f("ck_invoices_currency_iso4217")),
        sa.CheckConstraint(
            "status IN ('pending', 'paid', 'cancelled')", name=op.f("ck_invoices_status_valid")
        ),
        sa.CheckConstraint("amount >= 0", name=op.f("ck_invoices_amount_non_negative")),
        sa.CheckConstraint("due_at >= issued_at", name=op.f("ck_invoices_due_after_issue")),
        sa.ForeignKeyConstraint(
            ["tenant_id", "order_id"],
            ["orders.tenant_id", "orders.id"],
            name=op.f("fk_invoices_tenant_id_order_id_orders"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_invoices")),
        sa.UniqueConstraint(
            "tenant_id", "invoice_number", name=op.f("uq_invoices_tenant_id_invoice_number")
        ),
    )
    op.create_index(
        "ix_invoices_tenant_id_due_at", "invoices", ["tenant_id", "due_at"], unique=False
    )
    op.create_index(
        "ix_invoices_tenant_id_order_id", "invoices", ["tenant_id", "order_id"], unique=False
    )
    op.create_index(
        "ix_invoices_tenant_id_status_due_at",
        "invoices",
        ["tenant_id", "status", "due_at"],
        unique=False,
    )
    op.create_table(
        "order_items",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("order_id", sa.Uuid(), nullable=False),
        sa.Column("product_id", sa.Uuid(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("unit_price", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("line_total", sa.Numeric(precision=12, scale=2), nullable=False),
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
            "line_total = quantity * unit_price", name=op.f("ck_order_items_line_total_matches")
        ),
        sa.CheckConstraint("quantity > 0", name=op.f("ck_order_items_quantity_positive")),
        sa.CheckConstraint("unit_price >= 0", name=op.f("ck_order_items_unit_price_non_negative")),
        sa.ForeignKeyConstraint(
            ["tenant_id", "order_id"],
            ["orders.tenant_id", "orders.id"],
            name=op.f("fk_order_items_tenant_id_order_id_orders"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "product_id"],
            ["products.tenant_id", "products.id"],
            name=op.f("fk_order_items_tenant_id_product_id_products"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_order_items")),
    )
    op.create_index(
        "ix_order_items_tenant_id_order_id",
        "order_items",
        ["tenant_id", "order_id"],
        unique=False,
    )
    op.create_index(
        "ix_order_items_tenant_id_product_id",
        "order_items",
        ["tenant_id", "product_id"],
        unique=False,
    )
    op.create_table(
        "shipments",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("order_id", sa.Uuid(), nullable=False),
        sa.Column("shipment_number", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("carrier", sa.String(length=64), nullable=False),
        sa.Column("tracking_number", sa.String(length=64), nullable=True),
        sa.Column("shipped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expected_delivery_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delay_reason", sa.Text(), nullable=True),
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
            "delivered_at IS NULL OR status IN ('delivered', 'returned')",
            name=op.f("ck_shipments_delivered_at_only_when_delivered_or_returned"),
        ),
        sa.CheckConstraint(
            "status <> 'delivered' OR delivered_at IS NOT NULL",
            name=op.f("ck_shipments_delivered_requires_delivered_at"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'in_transit', 'delayed', 'delivered', 'returned')",
            name=op.f("ck_shipments_status_valid"),
        ),
        sa.CheckConstraint(
            "delivered_at IS NULL OR shipped_at IS NULL OR delivered_at >= shipped_at",
            name=op.f("ck_shipments_delivered_after_shipped"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "order_id"],
            ["orders.tenant_id", "orders.id"],
            name=op.f("fk_shipments_tenant_id_order_id_orders"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_shipments")),
        sa.UniqueConstraint(
            "tenant_id", "shipment_number", name=op.f("uq_shipments_tenant_id_shipment_number")
        ),
    )
    op.create_index(
        "ix_shipments_tenant_id_order_id", "shipments", ["tenant_id", "order_id"], unique=False
    )
    op.create_index(
        "ix_shipments_tenant_id_status", "shipments", ["tenant_id", "status"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_shipments_tenant_id_status", table_name="shipments")
    op.drop_index("ix_shipments_tenant_id_order_id", table_name="shipments")
    op.drop_table("shipments")
    op.drop_index("ix_order_items_tenant_id_product_id", table_name="order_items")
    op.drop_index("ix_order_items_tenant_id_order_id", table_name="order_items")
    op.drop_table("order_items")
    op.drop_index("ix_invoices_tenant_id_status_due_at", table_name="invoices")
    op.drop_index("ix_invoices_tenant_id_order_id", table_name="invoices")
    op.drop_index("ix_invoices_tenant_id_due_at", table_name="invoices")
    op.drop_table("invoices")
    op.drop_index("ix_orders_tenant_id_status", table_name="orders")
    op.drop_index("ix_orders_tenant_id_placed_at", table_name="orders")
    op.drop_index("ix_orders_tenant_id_customer_id_placed_at", table_name="orders")
    op.drop_table("orders")
    op.drop_index("ix_products_tenant_id_name", table_name="products")
    op.drop_table("products")
    op.drop_index("ix_customers_tenant_id_name", table_name="customers")
    op.drop_index("ix_customers_tenant_id_email", table_name="customers")
    op.drop_table("customers")
    op.drop_table("tenants")
