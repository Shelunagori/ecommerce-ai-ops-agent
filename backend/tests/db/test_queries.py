"""Service-level behaviour and tenant isolation of the query layer (real PostgreSQL)."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import InternalError

from app.core.errors import NotFoundError
from app.core.tenant import TenantContext
from app.db import session as db_session_module
from app.models import Invoice, Order, OrderItem, Product
from app.models.enums import OrderStatus
from app.services import CommerceQueries
from app.services.base import TenantScopedQueries
from tests.db.conftest import fixed_clock


@pytest.fixture
def qa(db_session, tenant_a):
    return CommerceQueries(db_session, tenant_a, fixed_clock)


@pytest.fixture
def qb(db_session, tenant_b):
    return CommerceQueries(db_session, tenant_b, fixed_clock)


# --- same human-readable ids in both tenants -------------------------------------------
def test_same_order_number_resolves_per_tenant(qa, qb):
    a, b = qa.orders.get_by_number("ORD-1001"), qb.orders.get_by_number("ORD-1001")
    assert (a.currency, a.total_amount) == ("USD", Decimal("179.89"))
    assert (b.currency, b.total_amount) == ("EUR", Decimal("248.00"))
    assert {i.sku for i in a.items} == {"SKU-1001", "SKU-1005"}
    assert {i.product_name for i in b.items} == {
        "Linen Bedding Set (Queen)",
        "Organic Cotton Throw Blanket",
    }


@pytest.mark.parametrize(
    ("lookup", "ref"),
    [
        (lambda q, r: q.customers.get_by_code(r), "CUS-1001"),
        (lambda q, r: q.products.get_by_sku(r), "SKU-1001"),
        (lambda q, r: q.invoices.get_by_number(r), "INV-1001"),
        (lambda q, r: q.shipments.get_by_number(r), "SHP-1001"),
    ],
)
def test_shared_references_return_each_tenants_own_row(qa, qb, lookup, ref):
    assert lookup(qa, ref) != lookup(qb, ref)


@pytest.mark.parametrize(
    ("lookup", "ref", "resource"),
    [
        (lambda q, r: q.customers.get_by_code(r), "CUS-1005", "customer"),
        (lambda q, r: q.products.get_by_sku(r), "SKU-1005", "product"),
        (lambda q, r: q.orders.get_by_number(r), "ORD-1006", "order"),
        (lambda q, r: q.invoices.get_by_number(r), "INV-1005", "invoice"),
        (lambda q, r: q.shipments.get_by_number(r), "SHP-1005", "shipment"),
        (lambda q, r: q.orders.list_for_customer(r), "CUS-1005", "customer"),
        (lambda q, r: q.invoices.list_unpaid_for_customer(r), "CUS-1006", "customer"),
        (lambda q, r: q.shipments.list_for_order(r), "ORD-1006", "order"),
    ],
)
def test_tenant_b_cannot_reach_tenant_a_only_rows(qa, qb, lookup, ref, resource):
    lookup(qa, ref)  # exists in tenant A ...
    with pytest.raises(NotFoundError) as exc:  # ... but is invisible to tenant B
        lookup(qb, ref)
    assert exc.value.resource == resource


def test_unknown_tenant_sees_nothing(db_session):
    q = CommerceQueries(db_session, TenantContext(uuid.uuid4()), fixed_clock)
    assert q.customers.list_all() == []
    assert q.products.list_all() == []
    assert q.shipments.list_delayed() == []
    assert q.invoices.list_overdue() == []
    with pytest.raises(NotFoundError):
        q.orders.get_by_number("ORD-1001")


def test_query_classes_require_a_tenant_context(db_session, tenant_a):
    with pytest.raises(TypeError):
        TenantScopedQueries(db_session, tenant_a.tenant_id)  # type: ignore[arg-type]


# --- customers / products ----------------------------------------------------------------
def test_customer_search_is_case_insensitive_and_tenant_scoped(qa, qb):
    assert [c.customer_code for c in qa.customers.search_by_name("THOMPSON")] == [
        "CUS-1001",
        "CUS-1006",
    ]
    assert qb.customers.search_by_name("thompson") == []
    assert [c.name for c in qb.customers.search_by_name("chlo")] == ["Chloé Dubois"]


@pytest.mark.parametrize("term", ["%", "_", "\\"])
def test_search_wildcards_are_literal(qa, term):
    assert qa.customers.search_by_name(term) == []


def test_customer_list_and_pagination(qa, qb):
    assert len(qa.customers.list_all()) == 6
    assert len(qb.customers.list_all()) == 4
    page = qa.customers.list_all(limit=2, offset=2)
    assert [c.customer_code for c in page] == ["CUS-1003", "CUS-1004"]
    assert len(qa.customers.list_all(limit=10_000)) == 6  # limit is clamped, not an error


def test_product_lookup_and_search(qa, qb):
    assert qa.products.get_by_sku("SKU-1006").active is False
    assert [p.sku for p in qa.products.search_by_name("watch")] == ["SKU-1002"]
    assert qb.products.search_by_name("watch") == []


# --- orders ------------------------------------------------------------------------------
def test_customer_order_history_newest_first(qa):
    assert [o.order_number for o in qa.orders.list_for_customer("CUS-1001")] == [
        "ORD-1004",
        "ORD-1002",
        "ORD-1001",
    ]


def test_latest_order_for_customer(qa, qb):
    assert qa.orders.latest_for_customer("CUS-1001").order_number == "ORD-1004"
    assert qb.orders.latest_for_customer("CUS-1001").order_number == "ORD-1003"
    assert qa.orders.latest_for_customer("CUS-1005") is None  # customer exists, no orders


def test_orders_by_status(qa, qb):
    assert [o.order_number for o in qa.orders.list_by_status(OrderStatus.PROCESSING)] == [
        "ORD-1004"
    ]
    assert [o.order_number for o in qb.orders.list_by_status(OrderStatus.PROCESSING)] == [
        "ORD-1003"
    ]


def test_order_detail_returns_every_line_for_a_repeated_product(db_session, qa, tenant_a):
    order = qa.orders.get_by_number("ORD-1007")
    assert [(i.sku, i.quantity) for i in order.items] == [("SKU-1003", 2)]
    order_id = db_session.scalar(
        select(Order.id).where(
            Order.tenant_id == tenant_a.tenant_id, Order.order_number == "ORD-1007"
        )
    )
    product_id = db_session.scalar(
        select(Product.id).where(Product.tenant_id == tenant_a.tenant_id, Product.sku == "SKU-1003")
    )
    db_session.add(
        OrderItem(
            tenant_id=tenant_a.tenant_id,
            order_id=order_id,
            product_id=product_id,
            quantity=1,
            unit_price=Decimal("39.60"),  # e.g. a discounted second line
            line_total=Decimal("39.60"),
        )
    )
    db_session.flush()
    lines = qa.orders.get_by_number("ORD-1007").items
    assert sorted((i.sku, i.quantity, i.unit_price) for i in lines) == [
        ("SKU-1003", 1, Decimal("39.60")),
        ("SKU-1003", 2, Decimal("49.50")),
    ]


def test_seeded_totals_match_line_items_and_invoices(db_session):
    """E4: totals are kept consistent by the writer (seed), not by a trigger."""
    item_sums = (
        select(OrderItem.order_id, func.sum(OrderItem.line_total).label("s"))
        .group_by(OrderItem.order_id)
        .subquery()
    )
    mismatches = db_session.execute(
        select(Order.order_number)
        .join(item_sums, item_sums.c.order_id == Order.id)
        .where(item_sums.c.s != Order.total_amount)
    ).all()
    assert mismatches == []
    invoice_mismatch = db_session.execute(
        select(Invoice.invoice_number)
        .join(Order, Order.id == Invoice.order_id)
        .where(Invoice.amount != Order.total_amount)
    ).all()
    assert invoice_mismatch == []


# --- invoices ----------------------------------------------------------------------------
def test_unpaid_means_pending_regardless_of_due_date(qa):
    unpaid = qa.invoices.list_unpaid_for_customer("CUS-1001")
    assert [(i.invoice_number, i.is_overdue) for i in unpaid] == [
        ("INV-1002", True),  # past due -> derived overdue
        ("INV-1004", False),  # pending, not yet due
    ]


def test_latest_unpaid_invoice(qa):
    assert qa.invoices.latest_unpaid_for_customer("CUS-1001").invoice_number == "INV-1004"
    assert qa.invoices.latest_unpaid_for_customer("CUS-1003") is None


def test_overdue_is_derived_from_due_date(db_session, tenant_a, tenant_b):
    at_now = CommerceQueries(db_session, tenant_a, fixed_clock)
    assert [i.invoice_number for i in at_now.invoices.list_overdue()] == ["INV-1002", "INV-1007"]
    assert at_now.invoices.count_overdue() == 2
    earlier = CommerceQueries(db_session, tenant_a, lambda: datetime(2026, 8, 1, tzinfo=UTC))
    assert earlier.invoices.list_overdue() == []
    b = CommerceQueries(db_session, tenant_b, fixed_clock)
    assert [i.invoice_number for i in b.invoices.list_overdue()] == ["INV-1002"]


def test_paid_invoice_is_never_overdue(qa):
    inv = qa.invoices.get_by_number("INV-1001")
    assert inv.status == "paid" and inv.paid_at is not None and inv.is_overdue is False


# --- shipments ---------------------------------------------------------------------------
def test_delayed_shipments_per_tenant(qa, qb):
    [a] = qa.shipments.list_delayed()
    [b] = qb.shipments.list_delayed()
    assert (a.shipment_number, a.order_number) == ("SHP-1003", "ORD-1003")
    assert a.delay_reason == "Carrier hub closure due to severe weather"
    assert (b.shipment_number, b.delay_reason) == ("SHP-1002", None)  # reason not known yet


def test_shipments_for_order_and_latest(qa):
    assert [s.shipment_number for s in qa.shipments.list_for_order("ORD-1001")] == ["SHP-1001"]
    assert qa.shipments.latest_for_order("ORD-1007") is None
    assert qa.shipments.latest_for_customer("CUS-1001").shipment_number == "SHP-1004"


def test_demo_summary(qa, qb):
    a, b = qa.summary.demo_summary(), qb.summary.demo_summary()
    assert (a.tenant.slug, a.customers, a.orders, a.unpaid_invoices, a.overdue_invoices) == (
        "northstar-commerce",
        6,
        10,
        3,
        2,
    )
    assert (b.tenant.slug, b.customers, b.orders, b.unpaid_invoices, b.overdue_invoices) == (
        "bluepeak-retail",
        4,
        5,
        2,
        1,
    )
    assert a.delayed_shipments == b.delayed_shipments == 1


# --- transaction semantics ---------------------------------------------------------------
def test_read_only_session_rejects_writes(db_engine, monkeypatch, tenant_a):
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(bind=db_engine)
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    with db_session_module.read_only_session() as session:
        assert CommerceQueries(session, tenant_a).customers.get_by_code("CUS-1001")
        with pytest.raises(InternalError, match="read-only transaction"):
            session.execute(Order.__table__.delete())


def test_queries_never_commit(db_session, qa):
    """Query classes leave the caller's transaction untouched."""
    qa.customers.list_all(limit=1)  # autobegin
    before = db_session.get_transaction()
    assert before is not None
    qa.orders.get_by_number("ORD-1001")
    qa.invoices.list_overdue()
    qa.summary.demo_summary()
    assert db_session.get_transaction() is before
    assert not db_session.new and not db_session.dirty
