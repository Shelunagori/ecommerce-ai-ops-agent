"""SUPPLEMENTARY regression guard (not the primary proof of isolation).

Calls every public query method and checks that every SQL statement it emits carries the
current tenant id as a bound parameter. The authoritative guarantees are the composite
FKs and the service/API isolation tests; this only catches a future query that forgets
the tenant filter entirely.
"""

import inspect as pyinspect

import pytest
from sqlalchemy import event

from app.core.errors import NotFoundError
from app.models.enums import OrderStatus, ShipmentStatus
from app.services import (
    CustomerQueries,
    InvoiceQueries,
    OrderQueries,
    ProductQueries,
    ShipmentQueries,
    SummaryQueries,
)
from tests.db.conftest import fixed_clock

SAMPLE_ARGS = {
    "customer_code": "CUS-1001",
    "order_number": "ORD-1001",
    "invoice_number": "INV-1001",
    "shipment_number": "SHP-1001",
    "sku": "SKU-1001",
    "term": "a",
    "status": None,
}
STATUS_FOR = {OrderQueries: OrderStatus.PROCESSING, ShipmentQueries: ShipmentStatus.DELAYED}
CLASSES = [
    CustomerQueries,
    ProductQueries,
    OrderQueries,
    InvoiceQueries,
    ShipmentQueries,
    SummaryQueries,
]


def _public_methods(cls):
    return [
        (name, fn)
        for name, fn in pyinspect.getmembers(cls, pyinspect.isfunction)
        if not name.startswith("_")
    ]


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_every_statement_is_bound_to_the_tenant(db_session, tenant_a, cls):
    captured: list[tuple[str, object]] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        captured.append((statement, parameters))

    engine = db_session.get_bind().engine
    event.listen(engine, "before_cursor_execute", capture)
    try:
        queries = cls(db_session, tenant_a, fixed_clock)
        methods = _public_methods(cls)
        assert methods, f"{cls.__name__} has no public methods?"
        for name, fn in methods:
            params = list(pyinspect.signature(fn).parameters)[1:]
            kwargs = {
                p: (STATUS_FOR.get(cls) if p == "status" else SAMPLE_ARGS[p])
                for p in params
                if p in SAMPLE_ARGS
            }
            captured.clear()
            try:
                getattr(queries, name)(**kwargs)
            except NotFoundError:
                pass
            statements = [(s, p) for s, p in captured if s.lstrip().upper().startswith("SELECT")]
            assert statements, f"{cls.__name__}.{name} ran no SELECT"
            for statement, parameters in statements:
                values = parameters.values() if isinstance(parameters, dict) else parameters
                assert tenant_a.tenant_id in list(values), (
                    f"{cls.__name__}.{name} emitted a query without the tenant id:\n{statement}"
                )
    finally:
        event.remove(engine, "before_cursor_execute", capture)
