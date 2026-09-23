"""Agent tools against real PostgreSQL: correctness, tenant isolation, read-only execution,
per-call sessions, and the developer CLI."""

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import sessionmaker

import app.db.session as db_session_module
from app.agent.context import AgentContext, create_agent_context
from app.agent.tools import ToolDependencies, build_commerce_tools
from app.agent.tools.invoke import invoke_tool, make_runtime
from app.agent.tools.runtime import execute
from app.core.errors import TenantNotFoundError
from app.models import Customer
from scripts import run_tool
from tests.db.conftest import fixed_clock


@pytest.fixture
def factory(db_engine):
    return sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def deps(factory, monkeypatch):
    # The REAL read_only_session (SET TRANSACTION READ ONLY + rollback), bound to the test DB.
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    return ToolDependencies(session_scope=db_session_module.read_only_session, clock=fixed_clock)


@pytest.fixture
def tools(deps):
    return {t.name: t for t in build_commerce_tools(deps)}


@pytest.fixture
def ctx_a(tenant_a):
    return AgentContext(tenant_a.tenant_id, "req-a")


@pytest.fixture
def ctx_b(tenant_b):
    return AgentContext(tenant_b.tenant_id, "req-b")


def call(tools, name, ctx, **args):
    return invoke_tool(tools[name], args, ctx)


def data(result):
    assert result["ok"] is True, result
    return result["data"]


# --- correctness -------------------------------------------------------------------------------
def test_get_order_is_tenant_specific(tools, ctx_a, ctx_b):
    a = data(call(tools, "get_order", ctx_a, order_number="ORD-1001"))
    b = data(call(tools, "get_order", ctx_b, order_number="ORD-1001"))
    assert (a["currency"], a["total_amount"]) == ("USD", "179.89")
    assert (b["currency"], b["total_amount"]) == ("EUR", "248.00")
    assert [i["product_name"] for i in a["items"]] == [
        "Wireless Earbuds Pro",
        "Stainless Water Bottle",
    ]
    assert a["placed_at"] == "2026-06-10T09:15:00Z"  # ISO-8601


def test_repeated_product_order_returns_two_distinct_lines(tools, ctx_a):
    order = data(call(tools, "get_order", ctx_a, order_number="ORD-1010"))
    assert [
        (i["sku"], i["quantity"], i["unit_price"], i["line_total"]) for i in order["items"]
    ] == [
        ("SKU-1005", 4, "24.95", "99.80"),
        ("SKU-1005", 2, "19.96", "39.92"),
    ]
    assert order["total_amount"] == "139.72"


def test_latest_unpaid_invoice_and_derived_overdue(tools, ctx_a, ctx_b):
    inv = data(call(tools, "get_latest_unpaid_invoice", ctx_a, customer_code="CUS-1001"))
    assert (inv["invoice_number"], inv["status"], inv["amount"], inv["currency"]) == (
        "INV-1004",
        "pending",
        "223.95",
        "USD",
    )
    assert inv["due_at"] == "2026-10-18T19:10:00Z" and inv["is_overdue"] is False
    overdue = data(call(tools, "get_invoice", ctx_a, invoice_number="INV-1002"))
    assert (overdue["status"], overdue["is_overdue"]) == ("pending", True)
    paid = data(call(tools, "get_invoice", ctx_a, invoice_number="INV-1001"))
    assert (paid["status"], paid["is_overdue"]) == ("paid", False)
    assert (
        data(call(tools, "get_latest_unpaid_invoice", ctx_b, customer_code="CUS-1002"))[
            "invoice_number"
        ]
        == "INV-1002"
    )


def test_no_applicable_result_is_ok_with_null_data(tools, ctx_a):
    assert call(tools, "get_latest_unpaid_invoice", ctx_a, customer_code="CUS-1003") == {
        "ok": True,
        "data": None,
    }
    assert call(tools, "get_latest_customer_order", ctx_a, customer_code="CUS-1005") == {
        "ok": True,
        "data": None,
    }


def test_delayed_shipments(tools, ctx_a, ctx_b):
    a = data(call(tools, "list_delayed_shipments", ctx_a))
    b = data(call(tools, "list_delayed_shipments", ctx_b))
    assert [(s["shipment_number"], s["status"]) for s in a["items"]] == [("SHP-1003", "delayed")]
    assert a["items"][0]["delay_reason"] == "Carrier hub closure due to severe weather"
    assert [(s["shipment_number"], s["delay_reason"]) for s in b["items"]] == [("SHP-1002", None)]
    assert (a["count"], a["has_more"]) == (1, False)


def test_order_shipments_and_shipment_lookup(tools, ctx_a):
    ships = data(call(tools, "get_order_shipments", ctx_a, order_number="ORD-1001"))
    assert [s["shipment_number"] for s in ships["items"]] == ["SHP-1001"]
    assert data(call(tools, "get_order_shipments", ctx_a, order_number="ORD-1007"))["items"] == []
    shp = data(call(tools, "get_shipment", ctx_a, shipment_number="SHP-1001"))
    assert (shp["carrier"], shp["status"], shp["order_number"]) == ("UPS", "delivered", "ORD-1001")


def test_customer_tools(tools, ctx_a, ctx_b):
    found = data(call(tools, "search_customers", ctx_a, query="THOMPSON"))
    assert [c["customer_code"] for c in found["items"]] == ["CUS-1001", "CUS-1006"]
    assert data(call(tools, "search_customers", ctx_b, query="thompson"))["items"] == []
    page = data(call(tools, "search_customers", ctx_a, query="a", limit=2))
    assert (page["count"], page["has_more"]) == (2, True)
    assert (
        data(call(tools, "get_customer", ctx_b, customer_code="CUS-1001"))["name"]
        == "Emma Schneider"
    )
    assert data(call(tools, "search_customers", ctx_a, query="%"))["items"] == []


def test_order_list_tools(tools, ctx_a):
    orders = data(call(tools, "list_customer_orders", ctx_a, customer_code="CUS-1001"))
    assert [o["order_number"] for o in orders["items"]] == ["ORD-1004", "ORD-1002", "ORD-1001"]
    assert "items" not in orders["items"][0]  # summaries only
    latest = data(call(tools, "get_latest_customer_order", ctx_a, customer_code="CUS-1001"))
    assert latest["order_number"] == "ORD-1004"


def test_product_tools(tools, ctx_a, ctx_b):
    assert data(call(tools, "get_product", ctx_a, sku="SKU-1006"))["active"] is False
    assert data(call(tools, "get_product", ctx_a, sku="SKU-1005"))["unit_price"] == "24.95"
    assert [
        p["sku"] for p in data(call(tools, "search_products", ctx_a, query="watch"))["items"]
    ] == ["SKU-1002"]
    assert data(call(tools, "search_products", ctx_b, query="watch"))["items"] == []


def test_whitespace_around_identifiers_is_trimmed(tools, ctx_a):
    assert (
        data(call(tools, "get_order", ctx_a, order_number="  ORD-1001 "))["order_number"]
        == "ORD-1001"
    )


# --- structured result contract ------------------------------------------------------------------
def test_tools_return_structured_objects(tools, ctx_a):
    """Direct invocation returns the envelope as a plain JSON-compatible dict, not a string."""
    runtime = make_runtime(ctx_a)
    ok = tools["get_invoice"].invoke({"invoice_number": "INV-1002", "runtime": runtime})
    assert isinstance(ok, dict) and ok["ok"] is True
    inv = ok["data"]
    assert inv["amount"] == "49.50" and isinstance(inv["amount"], str)  # money as string
    assert inv["due_at"] == "2026-08-21T16:40:00Z"  # ISO-8601
    assert inv["paid_at"] is None and inv["is_overdue"] is True
    assert json.loads(json.dumps(ok)) == ok  # JSON-compatible, no Decimal/datetime objects

    missing = tools["get_order"].invoke({"order_number": "ORD-9999", "runtime": runtime})
    assert missing == {
        "ok": False,
        "error": {"code": "order_not_found", "message": "Order 'ORD-9999' was not found."},
    }

    page = tools["search_products"].invoke({"query": "a", "limit": 2, "runtime": runtime})
    assert isinstance(page, dict) and set(page["data"]) == {"items", "count", "has_more"}


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("get_order", {"order_number": "ORD-1010"}),
        ("get_shipment", {"shipment_number": "SHP-9999"}),
    ],
)
def test_tool_call_produces_model_consumable_message(tools, ctx_a, name, args):
    """As an agent runtime would call it: a ToolCall in, a ToolMessage with JSON text out."""
    direct = tools[name].invoke({**args, "runtime": make_runtime(ctx_a)})
    message = tools[name].invoke(
        {
            "type": "tool_call",
            "id": "call-7",
            "name": name,
            "args": {**args, "runtime": make_runtime(ctx_a)},
        }
    )
    assert message.tool_call_id == "call-7" and isinstance(message.content, str)
    assert json.loads(message.content) == direct
    assert str(ctx_a.tenant_id) not in message.content


# --- tenant isolation ----------------------------------------------------------------------------
# (tool, argument, reference existing ONLY in Northstar, resource)
A_ONLY = [
    ("get_customer", "customer_code", "CUS-1005", "customer"),
    ("list_customer_orders", "customer_code", "CUS-1005", "customer"),
    ("get_latest_customer_order", "customer_code", "CUS-1006", "customer"),
    ("get_latest_unpaid_invoice", "customer_code", "CUS-1006", "customer"),
    ("get_order", "order_number", "ORD-1006", "order"),
    ("get_order_shipments", "order_number", "ORD-1006", "order"),
    ("get_invoice", "invoice_number", "INV-1005", "invoice"),
    ("get_shipment", "shipment_number", "SHP-1005", "shipment"),
    ("get_product", "sku", "SKU-1005", "product"),
]


def _normalise(result, ref):
    return json.loads(json.dumps(result).replace(ref, "<REF>"))


@pytest.mark.parametrize(("name", "arg", "ref", "resource"), A_ONLY)
def test_other_tenants_reference_looks_exactly_like_a_missing_one(
    tools, ctx_a, ctx_b, name, arg, ref, resource
):
    assert call(tools, name, ctx_a, **{arg: ref})["ok"] is True  # exists for Northstar
    cross = call(tools, name, ctx_b, **{arg: ref})  # BluePeak asks for it
    missing_ref = "ZZZ-9999"
    missing = call(tools, name, ctx_b, **{arg: missing_ref})
    assert cross["error"]["code"] == f"{resource}_not_found"
    assert _normalise(cross, ref) == _normalise(missing, missing_ref)


def test_tenant_a_never_sees_tenant_b_values(tools, ctx_a):
    blob = json.dumps(
        [
            call(tools, "get_customer", ctx_a, customer_code="CUS-1001"),
            call(tools, "get_order", ctx_a, order_number="ORD-1001"),
            call(tools, "search_products", ctx_a, query="e", limit=20),
            call(tools, "list_delayed_shipments", ctx_a, limit=20),
        ]
    )
    for b_value in ("Emma Schneider", "EUR", "Linen Bedding", "GLS", "example.org"):
        assert b_value not in blob


def test_unknown_tenant_context_sees_nothing(tools):
    ghost = AgentContext(uuid.uuid4())
    assert (
        call(tools, "get_order", ghost, order_number="ORD-1001")["error"]["code"]
        == "order_not_found"
    )
    assert data(call(tools, "list_delayed_shipments", ghost))["items"] == []


def test_context_creation_boundary_checks_tenant(factory, tenant_a):
    with factory() as session:
        assert (
            create_agent_context(session, tenant_a.tenant_id, "r1").tenant_id == tenant_a.tenant_id
        )
        with pytest.raises(TenantNotFoundError):
            create_agent_context(session, uuid.uuid4())


# --- session behaviour -------------------------------------------------------------------------
def test_tool_execution_cannot_mutate_the_database(deps, factory, ctx_a):
    with factory() as s:
        before = s.scalar(select(func.count()).select_from(Customer))

    def malicious(q):  # an operation that tries to write through its session
        q.customers._session.execute(delete(Customer))
        return None

    result = execute("get_customer", make_runtime(ctx_a), deps, malicious)
    assert result["error"]["code"] == "internal_error"
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(Customer)) == before


def test_each_call_gets_its_own_session(factory, monkeypatch, ctx_a):
    opened = []

    @contextmanager
    def tracking_scope():
        with factory() as session:
            opened.append(id(session))
            yield session

    tools = {
        t.name: t for t in build_commerce_tools(ToolDependencies(session_scope=tracking_scope))
    }
    for _ in range(3):
        call(tools, "get_order", ctx_a, order_number="ORD-1001")
    assert len(opened) == 3 and len(set(opened)) == 3


def test_concurrent_calls_stay_isolated(tools, ctx_a, ctx_b):
    jobs = [(ctx_a, "USD"), (ctx_b, "EUR")] * 10
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda j: call(tools, "get_order", j[0], order_number="ORD-1001"), jobs)
        )
    assert [r["data"]["currency"] for r in results] == [c for _, c in jobs]


# --- developer CLI ---------------------------------------------------------------------------
@pytest.fixture
def cli(factory, monkeypatch):
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    return run_tool.main


def test_cli_runs_registered_tool(cli, tenant_a, capsys):
    code = cli(
        [
            "--tenant",
            str(tenant_a.tenant_id),
            "--tool",
            "get_order",
            "--args",
            '{"order_number": "ORD-1010"}',
            "--request-id",
            "cli-test",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and len(out["data"]["items"]) == 2


def test_cli_rejects_tenant_in_args(cli, tenant_a, tenant_b, capsys):
    code = cli(
        [
            "--tenant",
            str(tenant_a.tenant_id),
            "--tool",
            "get_order",
            "--args",
            json.dumps({"order_number": "ORD-1001", "tenant_id": str(tenant_b.tenant_id)}),
        ]
    )
    assert code == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_arguments"


def test_cli_rejects_unknown_tenant_and_tool(cli, tenant_a, capsys):
    assert (
        cli(
            [
                "--tenant",
                str(uuid.uuid4()),
                "--tool",
                "get_order",
                "--args",
                '{"order_number": "ORD-1001"}',
            ]
        )
        == 2
    )
    with pytest.raises(SystemExit):
        cli(["--tenant", str(tenant_a.tenant_id), "--tool", "run_sql", "--args", "{}"])


def test_cli_list_shows_no_context_fields(cli, capsys):
    assert cli(["--list"]) == 0
    listing = capsys.readouterr().out.lower()
    assert "tenant" not in listing and "runtime" not in listing and "request_id" not in listing
