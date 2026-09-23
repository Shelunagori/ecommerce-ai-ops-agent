"""Seed idempotency and safety rules (real PostgreSQL)."""

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from app.models import Customer, Order, OrderItem, Product, Tenant
from scripts import seed_demo
from scripts.seed_demo import (
    DEMO_TENANTS,
    NORTHSTAR,
    SeedConflictError,
    TenantSeed,
    seed,
    tenant_id_for,
)
from tests.conftest import make_settings


def _counts(session):
    return {
        m.__tablename__: session.scalar(select(func.count()).select_from(m))
        for m in seed_demo.ORDERED_MODELS
    }


def test_reseeding_is_a_no_op(db_session):
    before = _counts(db_session)
    results = seed(db_session)
    assert _counts(db_session) == before
    assert all(r.inserted == 0 and r.updated == 0 for r in results.values())


def test_reseed_restores_drifted_demo_rows(db_session, tenant_a):
    db_session.execute(
        Customer.__table__.update()
        .where(Customer.tenant_id == tenant_a.tenant_id, Customer.customer_code == "CUS-1001")
        .values(name="Tampered")
    )
    results = seed(db_session)
    assert results["customers"].updated == 1
    name = db_session.scalar(
        select(Customer.name).where(
            Customer.tenant_id == tenant_a.tenant_id, Customer.customer_code == "CUS-1001"
        )
    )
    assert name == "Ava Thompson"


def test_seed_never_touches_unrelated_rows(db_session, tenant_a):
    other = Customer(
        tenant_id=tenant_a.tenant_id,
        customer_code="CUS-7777",
        name="Real-looking but unrelated",
        email="unrelated@example.com",
    )
    db_session.add(other)
    db_session.flush()
    seed(db_session)
    db_session.expire_all()
    assert db_session.get(Customer, other.id).name == "Real-looking but unrelated"


def test_seed_aborts_instead_of_overwriting_a_foreign_natural_key(db_session):
    """A non-demo row already holding a demo natural key makes the seed fail, not overwrite."""
    stranger = Tenant(id=uuid.uuid4(), name="Someone else", slug="conflict-store")
    db_session.add(stranger)
    db_session.flush()
    conflicting = TenantSeed(
        slug="conflict-store", name="Conflict Store", currency="USD", customers=(), products=()
    )
    with pytest.raises(SeedConflictError, match="uq_tenants_slug"), db_session.begin_nested():
        seed(db_session, (conflicting,))
    assert db_session.get(Tenant, stranger.id).name == "Someone else"


def test_demo_ids_are_deterministic():
    assert tenant_id_for("northstar-commerce") == uuid.UUID("17243d88-ed66-5445-955b-7d7572094122")
    assert tenant_id_for("bluepeak-retail") == uuid.UUID("11a6d918-f89f-59b5-ab1e-45a109978c8a")


def test_seed_covers_required_scenarios(db_session, tenant_a):
    assert {t.slug for t in DEMO_TENANTS} == {"northstar-commerce", "bluepeak-retail"}
    orders_for_cus_1001 = db_session.scalar(
        select(func.count())
        .select_from(Order)
        .join(Customer, Customer.id == Order.customer_id)
        .where(Order.tenant_id == tenant_a.tenant_id, Customer.customer_code == "CUS-1001")
    )
    assert orders_for_cus_1001 >= 3
    statuses = {o.status for o in NORTHSTAR.orders}
    assert {"delivered", "processing", "shipped"} <= statuses


def test_seed_refuses_production(monkeypatch, capsys):
    monkeypatch.setattr(seed_demo, "get_settings", lambda: make_settings(app_env="production"))
    assert seed_demo.main() == 2
    assert "production" in capsys.readouterr().err


def _lines(session, tenant_id, order_number):
    return session.execute(
        select(
            OrderItem.id,
            Product.sku,
            OrderItem.quantity,
            OrderItem.unit_price,
            OrderItem.line_total,
        )
        .join(Order, Order.id == OrderItem.order_id)
        .join(Product, Product.id == OrderItem.product_id)
        .where(Order.tenant_id == tenant_id, Order.order_number == order_number)
        .order_by(OrderItem.quantity.desc())
    ).all()


def test_seeded_order_repeats_a_product_on_two_distinct_lines(db_session, tenant_a):
    """ORD-1010 lists SKU-1005 twice (full price + promo line): two rows, two ids."""
    lines = _lines(db_session, tenant_a.tenant_id, "ORD-1010")
    assert [(sku, qty, price) for _, sku, qty, price, _ in lines] == [
        ("SKU-1005", 4, Decimal("24.95")),
        ("SKU-1005", 2, Decimal("19.96")),
    ]
    assert len({line_id for line_id, *_ in lines}) == 2
    total = db_session.scalar(
        select(Order.total_amount).where(
            Order.tenant_id == tenant_a.tenant_id, Order.order_number == "ORD-1010"
        )
    )
    assert total == sum(lt for *_, lt in lines) == Decimal("139.72")


def test_repeated_product_lines_survive_reseeding(db_session, tenant_a):
    before = _lines(db_session, tenant_a.tenant_id, "ORD-1010")
    assert len(before) == 2
    results = seed(db_session)
    assert results["order_items"].inserted == 0 and results["order_items"].updated == 0
    assert _lines(db_session, tenant_a.tenant_id, "ORD-1010") == before


def test_line_ids_do_not_depend_on_product():
    """Demo line identity is (tenant, order, line number) - never the SKU."""
    from scripts.seed_demo import build_rows

    items = build_rows(NORTHSTAR)[OrderItem]
    ids = [r["id"] for r in items]
    assert len(ids) == len(set(ids))
    assert seed_demo.demo_id(NORTHSTAR.slug, "order_item", "ORD-1010", "line-2") in ids
