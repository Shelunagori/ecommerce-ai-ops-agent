"""Idempotent development seed with entirely SYNTHETIC ecommerce data.

Usage (from backend/):  uv run python -m scripts.seed_demo

Safety rules:
* Refuses to run when APP_ENV=production.
* Only deterministic demo records are touched: every row id is a UUID5 derived from the
  tenant slug and the row's natural key, and upserts conflict on that id only.
* It NEVER deletes rows and NEVER overwrites rows it did not create: if an unrelated row
  already holds one of the demo natural keys (e.g. a slug or order number), the unique
  constraint fails and the whole seed rolls back without changing anything.
* Re-running converges known demo rows back to their canonical values; rows that are
  already canonical are left untouched (updated_at does not change).
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Customer, Invoice, Order, OrderItem, Product, Shipment, Tenant

# Fixed namespace: changing it would change every demo id.
DEMO_NAMESPACE = uuid.UUID("6f1c0d2e-3b7a-4c1e-9a64-0c0ffee0c0de")


def demo_id(*parts: str) -> uuid.UUID:
    return uuid.uuid5(DEMO_NAMESPACE, "/".join(parts))


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


# --------------------------------------------------------------------------- data
@dataclass(frozen=True)
class LineSeed:
    """One order line. ``line`` is its stable synthetic key within the order (never reuse or
    renumber); the line's demo id derives from (tenant, order, line) - not from the SKU,
    so the same product may appear on several lines."""

    line: int
    sku: str
    quantity: int
    unit_price: str | None = None  # None -> product list price


@dataclass(frozen=True)
class InvoiceSeed:
    number: str
    status: str
    issued_at: str
    due_in_days: int
    paid_at: str | None = None


@dataclass(frozen=True)
class ShipmentSeed:
    number: str
    status: str
    carrier: str
    tracking: str | None = None
    shipped_at: str | None = None
    expected_at: str | None = None
    delivered_at: str | None = None
    delay_reason: str | None = None


@dataclass(frozen=True)
class OrderSeed:
    number: str
    customer: str
    status: str
    placed_at: str | None
    items: tuple[LineSeed, ...]
    invoice: InvoiceSeed | None = None
    shipments: tuple[ShipmentSeed, ...] = ()


@dataclass(frozen=True)
class TenantSeed:
    slug: str
    name: str
    currency: str
    customers: tuple[tuple[str, str, str, str], ...]  # code, name, email, status
    products: tuple[tuple[str, str, str, str, bool], ...]  # sku, name, desc, price, active
    orders: tuple[OrderSeed, ...] = field(default=())


NORTHSTAR = TenantSeed(
    slug="northstar-commerce",
    name="Northstar Commerce",
    currency="USD",
    customers=(
        ("CUS-1001", "Ava Thompson", "ava.thompson@example.com", "active"),
        ("CUS-1002", "Liam Carter", "liam.carter@example.com", "active"),
        ("CUS-1003", "Sofia Martinez", "sofia.martinez@example.com", "active"),
        ("CUS-1004", "Noah Patel", "noah.patel@example.com", "active"),
        ("CUS-1005", "Mia Chen", "mia.chen@example.com", "inactive"),
        ("CUS-1006", "Avery Thompson-Lee", "avery.tl@example.com", "active"),
    ),
    products=(
        ("SKU-1001", "Wireless Earbuds Pro", "Bluetooth 5.3 earbuds, 30h battery.", "129.99", True),
        ("SKU-1002", "Smart Fitness Watch", "Heart-rate and GPS tracking.", "199.00", True),
        ("SKU-1003", "USB-C Charging Hub", "7-port hub with 100W passthrough.", "49.50", True),
        ("SKU-1004", "Ergonomic Office Chair", "Adjustable lumbar support.", "349.00", True),
        ("SKU-1005", "Stainless Water Bottle", "750ml, double-walled.", "24.95", True),
        ("SKU-1006", "Noise-Cancelling Headphones", "Discontinued model.", "279.00", False),
    ),
    orders=(
        # 1. Paid invoice + delivered shipment
        OrderSeed(
            "ORD-1001",
            "CUS-1001",
            "delivered",
            "2026-06-10T09:15:00",
            (LineSeed(1, "SKU-1001", 1), LineSeed(2, "SKU-1005", 2)),
            InvoiceSeed("INV-1001", "paid", "2026-06-10T09:15:00", 30, "2026-06-12T14:00:00"),
            (
                ShipmentSeed(
                    "SHP-1001",
                    "delivered",
                    "UPS",
                    "1Z999AA10000001",
                    "2026-06-11T08:00:00",
                    "2026-06-15T18:00:00",
                    "2026-06-14T11:20:00",
                ),
            ),
        ),
        # 2. Overdue invoice (pending, due date passed) - same customer, 2nd order
        OrderSeed(
            "ORD-1002",
            "CUS-1001",
            "delivered",
            "2026-07-22T16:40:00",
            (LineSeed(1, "SKU-1003", 1),),
            InvoiceSeed("INV-1002", "pending", "2026-07-22T16:40:00", 30),
            (
                ShipmentSeed(
                    "SHP-1002",
                    "delivered",
                    "USPS",
                    "9400100000000000000002",
                    "2026-07-23T10:00:00",
                    "2026-07-27T18:00:00",
                    "2026-07-26T15:05:00",
                ),
            ),
        ),
        # 3. Shipment delayed by several days, with a reason
        OrderSeed(
            "ORD-1003",
            "CUS-1002",
            "shipped",
            "2026-08-28T12:00:00",
            (LineSeed(1, "SKU-1004", 1),),
            InvoiceSeed("INV-1003", "paid", "2026-08-28T12:00:00", 14, "2026-08-28T12:05:00"),
            (
                ShipmentSeed(
                    "SHP-1003",
                    "delayed",
                    "FedEx",
                    "7700000000000003",
                    "2026-08-29T09:30:00",
                    "2026-09-02T18:00:00",
                    delay_reason="Carrier hub closure due to severe weather",
                ),
            ),
        ),
        # 4. Order still processing (invoice pending, not yet due; shipment pending)
        # 5. ...and CUS-1001's third order -> multiple historical orders
        OrderSeed(
            "ORD-1004",
            "CUS-1001",
            "processing",
            "2026-09-18T19:10:00",
            (LineSeed(1, "SKU-1002", 1), LineSeed(2, "SKU-1005", 1)),
            InvoiceSeed("INV-1004", "pending", "2026-09-18T19:10:00", 30),
            (ShipmentSeed("SHP-1004", "pending", "FedEx"),),
        ),
        OrderSeed(
            "ORD-1005",
            "CUS-1003",
            "delivered",
            "2026-08-05T08:45:00",
            (LineSeed(1, "SKU-1001", 2),),
            InvoiceSeed("INV-1005", "paid", "2026-08-05T08:45:00", 30, "2026-08-07T10:00:00"),
            (
                ShipmentSeed(
                    "SHP-1005",
                    "delivered",
                    "UPS",
                    "1Z999AA10000005",
                    "2026-08-06T08:00:00",
                    "2026-08-10T18:00:00",
                    "2026-08-09T13:30:00",
                ),
            ),
        ),
        OrderSeed(
            "ORD-1006",
            "CUS-1004",
            "cancelled",
            "2026-08-12T11:00:00",
            (LineSeed(1, "SKU-1002", 1),),
            InvoiceSeed("INV-1006", "cancelled", "2026-08-12T11:00:00", 30),
        ),
        OrderSeed(
            "ORD-1007",
            "CUS-1004",
            "confirmed",
            "2026-09-20T10:30:00",
            (LineSeed(1, "SKU-1003", 2),),
        ),
        OrderSeed("ORD-1008", "CUS-1003", "draft", None, (LineSeed(1, "SKU-1005", 3),)),
        # Same product on two separate lines: full-price line + promotional line.
        OrderSeed(
            "ORD-1010",
            "CUS-1006",
            "confirmed",
            "2026-09-22T09:00:00",
            (LineSeed(1, "SKU-1005", 4), LineSeed(2, "SKU-1005", 2, unit_price="19.96")),
        ),
        # Short payment terms -> overdue while its shipment is in transit
        OrderSeed(
            "ORD-1009",
            "CUS-1002",
            "shipped",
            "2026-09-15T15:00:00",
            (LineSeed(1, "SKU-1001", 1),),
            InvoiceSeed("INV-1007", "pending", "2026-09-15T15:00:00", 5),
            (
                ShipmentSeed(
                    "SHP-1006",
                    "in_transit",
                    "DHL",
                    "JD0000000000000006",
                    "2026-09-19T07:45:00",
                    "2026-09-25T18:00:00",
                ),
            ),
        ),
    ),
)

BLUEPEAK = TenantSeed(
    slug="bluepeak-retail",
    name="BluePeak Retail",
    currency="EUR",
    customers=(
        ("CUS-1001", "Emma Schneider", "emma.schneider@example.org", "active"),
        ("CUS-1002", "Lucas Moreau", "lucas.moreau@example.org", "active"),
        ("CUS-1003", "Chloé Dubois", "chloe.dubois@example.org", "active"),
        ("CUS-1004", "Mateo García", "mateo.garcia@example.org", "active"),
    ),
    products=(
        ("SKU-1001", "Organic Cotton Throw Blanket", "130x170cm, GOTS certified.", "59.00", True),
        (
            "SKU-1002",
            "Ceramic Pour-Over Coffee Set",
            "Dripper, carafe and two cups.",
            "42.50",
            True,
        ),
        ("SKU-1003", "Bamboo Cutting Board Set", "Three sizes, juice groove.", "34.90", True),
        ("SKU-1004", "Linen Bedding Set (Queen)", "Stonewashed French linen.", "189.00", True),
    ),
    orders=(
        OrderSeed(
            "ORD-1001",
            "CUS-1001",
            "delivered",
            "2026-07-02T10:00:00",
            (LineSeed(1, "SKU-1004", 1), LineSeed(2, "SKU-1001", 1)),
            InvoiceSeed("INV-1001", "paid", "2026-07-02T10:00:00", 14, "2026-07-03T09:00:00"),
            (
                ShipmentSeed(
                    "SHP-1001",
                    "delivered",
                    "DPD",
                    "DPD00000000001",
                    "2026-07-03T08:00:00",
                    "2026-07-07T18:00:00",
                    "2026-07-06T12:10:00",
                ),
            ),
        ),
        # Delayed shipment with NO reason yet (carrier has not reported one) + overdue invoice
        OrderSeed(
            "ORD-1002",
            "CUS-1002",
            "shipped",
            "2026-08-30T14:20:00",
            (LineSeed(1, "SKU-1002", 2),),
            InvoiceSeed("INV-1002", "pending", "2026-08-30T14:20:00", 15),
            (
                ShipmentSeed(
                    "SHP-1002",
                    "delayed",
                    "GLS",
                    "GLS00000000002",
                    "2026-08-31T09:00:00",
                    "2026-09-04T18:00:00",
                ),
            ),
        ),
        OrderSeed(
            "ORD-1003",
            "CUS-1001",
            "processing",
            "2026-09-17T17:45:00",
            (LineSeed(1, "SKU-1003", 1),),
            InvoiceSeed("INV-1003", "pending", "2026-09-17T17:45:00", 30),
            (ShipmentSeed("SHP-1003", "pending", "DPD"),),
        ),
        OrderSeed(
            "ORD-1004",
            "CUS-1003",
            "confirmed",
            "2026-09-21T08:30:00",
            (LineSeed(1, "SKU-1001", 2),),
        ),
        OrderSeed(
            "ORD-1005",
            "CUS-1004",
            "delivered",
            "2026-06-18T13:00:00",
            (LineSeed(1, "SKU-1002", 1),),
            InvoiceSeed("INV-1004", "paid", "2026-06-18T13:00:00", 14, "2026-06-19T08:00:00"),
            (
                ShipmentSeed(
                    "SHP-1004",
                    "returned",
                    "DHL",
                    "JD0000000000000104",
                    "2026-06-19T09:00:00",
                    "2026-06-23T18:00:00",
                    "2026-06-22T16:00:00",
                ),
            ),
        ),
    ),
)

DEMO_TENANTS: tuple[TenantSeed, ...] = (NORTHSTAR, BLUEPEAK)


def tenant_id_for(slug: str) -> uuid.UUID:
    return demo_id("tenant", slug)


# ------------------------------------------------------------------------ building rows
def build_rows(seed: TenantSeed) -> dict[type, list[dict[str, Any]]]:
    tid = tenant_id_for(seed.slug)
    rows: dict[type, list[dict[str, Any]]] = {
        Tenant: [{"id": tid, "name": seed.name, "slug": seed.slug}],
        Customer: [],
        Product: [],
        Order: [],
        OrderItem: [],
        Invoice: [],
        Shipment: [],
    }
    for code, name, email, status in seed.customers:
        rows[Customer].append(
            {
                "id": demo_id(seed.slug, "customer", code),
                "tenant_id": tid,
                "customer_code": code,
                "name": name,
                "email": email,
                "status": status,
            }
        )
    prices: dict[str, Decimal] = {}
    for sku, name, desc, price, active in seed.products:
        prices[sku] = Decimal(price)
        rows[Product].append(
            {
                "id": demo_id(seed.slug, "product", sku),
                "tenant_id": tid,
                "sku": sku,
                "name": name,
                "description": desc,
                "unit_price": Decimal(price),
                "currency": seed.currency,
                "active": active,
            }
        )
    for order in seed.orders:
        order_id = demo_id(seed.slug, "order", order.number)
        line_keys = [line.line for line in order.items]
        if len(line_keys) != len(set(line_keys)):
            raise ValueError(f"demo order {order.number} reuses a line number")
        total = Decimal("0.00")
        for line in order.items:
            unit_price = Decimal(line.unit_price) if line.unit_price else prices[line.sku]
            line_total = unit_price * line.quantity
            total += line_total
            rows[OrderItem].append(
                {
                    "id": demo_id(seed.slug, "order_item", order.number, f"line-{line.line}"),
                    "tenant_id": tid,
                    "order_id": order_id,
                    "product_id": demo_id(seed.slug, "product", line.sku),
                    "quantity": line.quantity,
                    "unit_price": unit_price,
                    "line_total": line_total,
                }
            )
        rows[Order].append(
            {
                "id": order_id,
                "tenant_id": tid,
                "customer_id": demo_id(seed.slug, "customer", order.customer),
                "order_number": order.number,
                "status": order.status,
                "currency": seed.currency,
                "total_amount": total,
                "placed_at": ts(order.placed_at) if order.placed_at else None,
            }
        )
        if inv := order.invoice:
            issued = ts(inv.issued_at)
            rows[Invoice].append(
                {
                    "id": demo_id(seed.slug, "invoice", inv.number),
                    "tenant_id": tid,
                    "order_id": order_id,
                    "invoice_number": inv.number,
                    "status": inv.status,
                    "currency": seed.currency,
                    "amount": total,
                    "issued_at": issued,
                    "due_at": issued + timedelta(days=inv.due_in_days),
                    "paid_at": ts(inv.paid_at) if inv.paid_at else None,
                }
            )
        for shp in order.shipments:
            rows[Shipment].append(
                {
                    "id": demo_id(seed.slug, "shipment", shp.number),
                    "tenant_id": tid,
                    "order_id": order_id,
                    "shipment_number": shp.number,
                    "status": shp.status,
                    "carrier": shp.carrier,
                    "tracking_number": shp.tracking,
                    "shipped_at": ts(shp.shipped_at) if shp.shipped_at else None,
                    "expected_delivery_at": ts(shp.expected_at) if shp.expected_at else None,
                    "delivered_at": ts(shp.delivered_at) if shp.delivered_at else None,
                    "delay_reason": shp.delay_reason,
                }
            )
    return rows


# ---------------------------------------------------------------------------- upserting
@dataclass
class TableResult:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0


class SeedConflictError(RuntimeError):
    """A non-demo row already uses one of the demo natural keys."""


def _upsert(session: Session, model: type, rows: list[dict[str, Any]]) -> TableResult:
    if not rows:
        return TableResult()
    table = model.__table__
    ids = [r["id"] for r in rows]
    existing = set(session.scalars(select(table.c.id).where(table.c.id.in_(ids))))
    stmt = pg_insert(table).values(rows)
    data_cols = [c for c in rows[0] if c != "id"]
    changed = or_(*(table.c[c].is_distinct_from(stmt.excluded[c]) for c in data_cols))
    stmt = stmt.on_conflict_do_update(
        index_elements=[table.c.id],  # only ever our own deterministic ids
        set_={**{c: stmt.excluded[c] for c in data_cols}, "updated_at": func.now()},
        where=changed,
    ).returning(table.c.id)
    touched = set(session.scalars(stmt))
    inserted = len(set(ids) - existing)
    updated = len(touched & existing)
    return TableResult(inserted, updated, len(ids) - inserted - updated)


ORDERED_MODELS = (Tenant, Customer, Product, Order, OrderItem, Invoice, Shipment)


def seed(
    session: Session, tenants: tuple[TenantSeed, ...] = DEMO_TENANTS
) -> dict[str, TableResult]:
    """Upsert the demo data into ``session``. Does not commit (caller owns the transaction)."""
    results = {m.__tablename__: TableResult() for m in ORDERED_MODELS}
    built = [build_rows(t) for t in tenants]
    try:
        for model in ORDERED_MODELS:
            rows = [r for b in built for r in b[model]]
            res = _upsert(session, model, rows)
            agg = results[model.__tablename__]
            agg.inserted += res.inserted
            agg.updated += res.updated
            agg.unchanged += res.unchanged
    except IntegrityError as exc:
        # Constraint name only - never the offending row values.
        constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        raise SeedConflictError(
            f"Seed aborted: an existing non-demo row conflicts with demo data "
            f"(constraint {constraint}). Nothing was changed."
        ) from None
    return results


def main() -> int:
    settings = get_settings()
    if settings.app_env == "production":
        print("Refusing to seed: APP_ENV=production.", file=sys.stderr)
        return 2
    from app.db.session import unit_of_work  # noqa: PLC0415 - needs DATABASE_URL only here

    try:
        with unit_of_work() as session:
            results = seed(session)
    except SeedConflictError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Synthetic demo data seeded (no real customer or company data).")
    for t in DEMO_TENANTS:
        print(f"  tenant {t.name:<20} slug={t.slug:<20} id={tenant_id_for(t.slug)}")
    print(f"  {'table':<12} {'inserted':>8} {'updated':>8} {'unchanged':>9}")
    for name, r in results.items():
        print(f"  {name:<12} {r.inserted:>8} {r.updated:>8} {r.unchanged:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
