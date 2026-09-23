"""Database-level guarantees: tenant-scoped uniqueness, tenant-aware FKs, CHECKs."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Customer, Invoice, Order, OrderItem, Product, Shipment, Tenant

T0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def violates(session: Session, constraint: str, *rows: object) -> None:
    """Assert that inserting ``rows`` fails on exactly the named constraint."""
    with pytest.raises(IntegrityError) as exc, session.begin_nested():
        session.add_all(rows)
        session.flush()
    assert exc.value.orig.diag.constraint_name == constraint


def ok(session: Session, *rows: object) -> None:
    with session.begin_nested():
        session.add_all(rows)
        session.flush()


def ids(session: Session, model: type, tenant_id: uuid.UUID, **where: str) -> uuid.UUID:
    stmt = select(model.id).where(model.tenant_id == tenant_id)
    for col, value in where.items():
        stmt = stmt.where(getattr(model, col) == value)
    return session.scalars(stmt).one()


def customer(tenant_id, code="CUS-9001", **kw):
    return Customer(
        tenant_id=tenant_id, customer_code=code, name="Test", email="t@example.com", **kw
    )


def product(tenant_id, sku="SKU-9001", **kw):
    kw.setdefault("unit_price", Decimal("10.00"))
    kw.setdefault("currency", "USD")
    return Product(tenant_id=tenant_id, sku=sku, name="Test", **kw)


def order(tenant_id, customer_id, number="ORD-9001", **kw):
    kw.setdefault("status", "confirmed")
    kw.setdefault("placed_at", T0)
    return Order(
        tenant_id=tenant_id,
        customer_id=customer_id,
        order_number=number,
        currency="USD",
        total_amount=Decimal("10.00"),
        **kw,
    )


def invoice(tenant_id, order_id, number="INV-9001", **kw):
    kw.setdefault("status", "pending")
    return Invoice(
        tenant_id=tenant_id,
        order_id=order_id,
        invoice_number=number,
        currency="USD",
        amount=Decimal("10.00"),
        issued_at=T0,
        due_at=T0 + timedelta(days=30),
        **kw,
    )


def shipment(tenant_id, order_id, number="SHP-9001", **kw):
    kw.setdefault("status", "pending")
    return Shipment(
        tenant_id=tenant_id, order_id=order_id, shipment_number=number, carrier="UPS", **kw
    )


# --- tenant-scoped uniqueness -------------------------------------------------------
def test_tenant_slug_is_globally_unique(db_session):
    violates(db_session, "uq_tenants_slug", Tenant(name="Dup", slug="northstar-commerce"))


def test_same_references_exist_in_both_tenants(db_session, tenant_a, tenant_b):
    for model, col, ref in [
        (Customer, "customer_code", "CUS-1001"),
        (Product, "sku", "SKU-1001"),
        (Order, "order_number", "ORD-1001"),
        (Invoice, "invoice_number", "INV-1001"),
        (Shipment, "shipment_number", "SHP-1001"),
    ]:
        a = ids(db_session, model, tenant_a.tenant_id, **{col: ref})
        b = ids(db_session, model, tenant_b.tenant_id, **{col: ref})
        assert a != b, f"{model.__name__} {ref} must be distinct rows per tenant"


def test_new_tenant_may_reuse_reference_values(db_session):
    t = Tenant(name="Third", slug="third-store")
    ok(db_session, t)
    ok(db_session, customer(t.id, "CUS-1001"), product(t.id, "SKU-1001"))


@pytest.mark.parametrize(
    ("factory", "constraint"),
    [
        (lambda s, t: customer(t, "CUS-1001"), "uq_customers_tenant_id_customer_code"),
        (lambda s, t: product(t, "SKU-1001"), "uq_products_tenant_id_sku"),
        (
            lambda s, t: order(t, ids(s, Customer, t, customer_code="CUS-1002"), "ORD-1001"),
            "uq_orders_tenant_id_order_number",
        ),
        (
            lambda s, t: invoice(t, ids(s, Order, t, order_number="ORD-1007"), "INV-1001"),
            "uq_invoices_tenant_id_invoice_number",
        ),
        (
            lambda s, t: shipment(t, ids(s, Order, t, order_number="ORD-1007"), "SHP-1001"),
            "uq_shipments_tenant_id_shipment_number",
        ),
    ],
)
def test_references_unique_within_tenant(db_session, tenant_a, factory, constraint):
    violates(db_session, constraint, factory(db_session, tenant_a.tenant_id))


# --- tenant-aware composite foreign keys --------------------------------------------
def test_order_cannot_reference_other_tenants_customer(db_session, tenant_a, tenant_b):
    b_customer = ids(db_session, Customer, tenant_b.tenant_id, customer_code="CUS-1001")
    violates(
        db_session,
        "fk_orders_tenant_id_customer_id_customers",
        order(tenant_a.tenant_id, b_customer),
    )


def test_order_item_cannot_reference_other_tenants_product(db_session, tenant_a, tenant_b):
    a_order = ids(db_session, Order, tenant_a.tenant_id, order_number="ORD-1007")
    b_product = ids(db_session, Product, tenant_b.tenant_id, sku="SKU-1004")
    violates(
        db_session,
        "fk_order_items_tenant_id_product_id_products",
        OrderItem(
            tenant_id=tenant_a.tenant_id,
            order_id=a_order,
            product_id=b_product,
            quantity=1,
            unit_price=Decimal("1.00"),
            line_total=Decimal("1.00"),
        ),
    )


def test_order_item_cannot_reference_other_tenants_order(db_session, tenant_a, tenant_b):
    b_order = ids(db_session, Order, tenant_b.tenant_id, order_number="ORD-1001")
    a_product = ids(db_session, Product, tenant_a.tenant_id, sku="SKU-1001")
    violates(
        db_session,
        "fk_order_items_tenant_id_order_id_orders",
        OrderItem(
            tenant_id=tenant_a.tenant_id,
            order_id=b_order,
            product_id=a_product,
            quantity=1,
            unit_price=Decimal("1.00"),
            line_total=Decimal("1.00"),
        ),
    )


@pytest.mark.parametrize(
    ("factory", "constraint"),
    [
        (invoice, "fk_invoices_tenant_id_order_id_orders"),
        (shipment, "fk_shipments_tenant_id_order_id_orders"),
    ],
)
def test_invoice_and_shipment_cannot_reference_other_tenants_order(
    db_session, tenant_a, tenant_b, factory, constraint
):
    b_order = ids(db_session, Order, tenant_b.tenant_id, order_number="ORD-1001")
    violates(db_session, constraint, factory(tenant_a.tenant_id, b_order))


def test_moving_an_order_to_another_tenant_is_rejected(db_session, tenant_a, tenant_b):
    """Re-pointing tenant_id alone breaks the composite FKs of the order and its children."""
    with pytest.raises(IntegrityError), db_session.begin_nested():
        db_session.execute(
            update(Order)
            .where(Order.tenant_id == tenant_a.tenant_id, Order.order_number == "ORD-1007")
            .values(tenant_id=tenant_b.tenant_id)
        )


# --- CHECK constraints ------------------------------------------------------------------
@pytest.fixture
def a_refs(db_session, tenant_a):
    t = tenant_a.tenant_id
    return {
        "t": t,
        "customer": ids(db_session, Customer, t, customer_code="CUS-1004"),
        "order": ids(db_session, Order, t, order_number="ORD-1007"),
        "product": ids(db_session, Product, t, sku="SKU-1004"),
    }


def item(r, quantity=1, unit_price="5.00", line_total=None):
    price = Decimal(unit_price)
    return OrderItem(
        tenant_id=r["t"],
        order_id=r["order"],
        product_id=r["product"],
        quantity=quantity,
        unit_price=price,
        line_total=Decimal(line_total) if line_total else price * quantity,
    )


def test_quantity_must_be_positive(db_session, a_refs):
    violates(db_session, "ck_order_items_quantity_positive", item(a_refs, quantity=0))


def test_line_total_must_equal_quantity_times_price(db_session, a_refs):
    violates(
        db_session,
        "ck_order_items_line_total_matches",
        item(a_refs, quantity=2, unit_price="5.00", line_total="9.99"),
    )


def test_same_product_may_appear_on_multiple_lines_of_an_order(db_session, a_refs):
    """Lines are identified by their own UUID, not by (order, product): the same product can
    appear on several lines (different price, discount, customisation, fulfilment, ...)."""
    first = item(a_refs, quantity=1, unit_price="5.00")
    second = item(a_refs, quantity=3, unit_price="4.50")
    ok(db_session, first, second)
    lines = db_session.scalars(
        select(OrderItem).where(
            OrderItem.order_id == a_refs["order"], OrderItem.product_id == a_refs["product"]
        )
    ).all()
    assert {line.id for line in lines} == {first.id, second.id}


def test_money_cannot_be_negative(db_session, a_refs):
    violates(
        db_session,
        "ck_products_unit_price_non_negative",
        product(a_refs["t"], unit_price=Decimal("-1.00")),
    )


def test_money_is_exact_decimal(db_session, a_refs):
    p = product(a_refs["t"], unit_price=Decimal("0.10"))
    ok(db_session, p)
    db_session.expire(p)
    assert p.unit_price * 3 == Decimal("0.30")
    assert isinstance(p.unit_price, Decimal)


@pytest.mark.parametrize(
    ("make", "constraint"),
    [
        (lambda r: customer(r["t"], status="blocked"), "ck_customers_status_valid"),
        (lambda r: order(r["t"], r["customer"], status="lost"), "ck_orders_status_valid"),
        (lambda r: invoice(r["t"], r["order"], status="overdue"), "ck_invoices_status_valid"),
        (lambda r: shipment(r["t"], r["order"], status="lost"), "ck_shipments_status_valid"),
        (lambda r: product(r["t"], currency="usd"), "ck_products_currency_iso4217"),
        (
            lambda r: order(r["t"], r["customer"], placed_at=None),
            "ck_orders_placed_at_unless_draft",
        ),
    ],
)
def test_status_and_format_checks(db_session, a_refs, make, constraint):
    violates(db_session, constraint, make(a_refs))


def test_overdue_is_not_a_storable_invoice_status(db_session, a_refs):
    """E3: overdue is derived from due_at, never persisted."""
    violates(
        db_session,
        "ck_invoices_status_valid",
        invoice(a_refs["t"], a_refs["order"], status="overdue"),
    )


def test_draft_order_may_have_no_placed_at(db_session, a_refs):
    ok(db_session, order(a_refs["t"], a_refs["customer"], status="draft", placed_at=None))


@pytest.mark.parametrize(
    ("kwargs", "allowed"),
    [
        ({"status": "paid", "paid_at": None}, False),
        ({"status": "paid", "paid_at": T0}, True),
        ({"status": "pending", "paid_at": T0}, False),
        ({"status": "cancelled", "paid_at": T0}, False),
        ({"status": "pending", "paid_at": None}, True),
    ],
)
def test_paid_at_present_iff_paid(db_session, a_refs, kwargs, allowed):
    row = invoice(a_refs["t"], a_refs["order"], **kwargs)
    if allowed:
        ok(db_session, row)
    else:
        violates(db_session, "ck_invoices_paid_at_iff_paid", row)


def test_due_date_not_before_issue(db_session, a_refs):
    row = invoice(a_refs["t"], a_refs["order"])
    row.due_at = T0 - timedelta(days=1)
    violates(db_session, "ck_invoices_due_after_issue", row)


def test_delivered_shipment_requires_delivered_at(db_session, a_refs):
    violates(
        db_session,
        "ck_shipments_delivered_requires_delivered_at",
        shipment(a_refs["t"], a_refs["order"], status="delivered", shipped_at=T0),
    )


def test_in_transit_shipment_cannot_carry_delivered_at(db_session, a_refs):
    violates(
        db_session,
        "ck_shipments_delivered_at_only_when_delivered_or_returned",
        shipment(a_refs["t"], a_refs["order"], status="in_transit", shipped_at=T0, delivered_at=T0),
    )


def test_delivery_cannot_precede_shipping(db_session, a_refs):
    violates(
        db_session,
        "ck_shipments_delivered_after_shipped",
        shipment(
            a_refs["t"],
            a_refs["order"],
            status="delivered",
            shipped_at=T0,
            delivered_at=T0 - timedelta(hours=1),
        ),
    )


def test_delayed_shipment_without_reason_is_allowed(db_session, a_refs):
    """E9: a shipment can be delayed before the carrier reports why."""
    ok(db_session, shipment(a_refs["t"], a_refs["order"], status="delayed", shipped_at=T0))


def test_updated_at_is_maintained_on_update(db_session, a_refs):
    c = customer(a_refs["t"])
    ok(db_session, c)
    old = datetime(2020, 1, 1, tzinfo=UTC)
    db_session.execute(update(Customer).where(Customer.id == c.id).values(updated_at=old))
    db_session.execute(update(Customer).where(Customer.id == c.id).values(name="Renamed"))
    refreshed = db_session.scalar(select(Customer.updated_at).where(Customer.id == c.id))
    assert refreshed > old
