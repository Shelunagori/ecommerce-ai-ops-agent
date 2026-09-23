"""API-level tenant context, isolation, 404s and error envelope (real PostgreSQL)."""

import uuid

import pytest


def assert_error(resp, status, code):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str)
    assert body["request_id"] == resp.headers["x-request-id"]
    return body


ROUTES = [
    "/api/customers",
    "/api/customers/CUS-1001",
    "/api/customers/CUS-1001/orders",
    "/api/customers/CUS-1001/invoices/unpaid",
    "/api/orders/ORD-1001",
    "/api/orders?status=processing",
    "/api/invoices/INV-1001",
    "/api/shipments/SHP-1001",
    "/api/shipments?status=delayed",
    "/api/products",
    "/api/products/SKU-1001",
    "/api/demo/summary",
]


# --- tenant context --------------------------------------------------------------------
@pytest.mark.parametrize("path", ROUTES)
def test_every_route_requires_tenant_header(api, path):
    assert_error(api.get(path), 400, "tenant_context_missing")


@pytest.mark.parametrize("value", ["not-a-uuid", "1234", "' OR 1=1 --"])
def test_malformed_tenant_header(api, value):
    assert_error(
        api.get("/api/customers", headers={"X-Tenant-ID": value}), 400, "tenant_context_invalid"
    )


def test_unknown_tenant_is_404(api):
    body = assert_error(
        api.get("/api/customers", headers={"X-Tenant-ID": str(uuid.uuid4())}),
        404,
        "tenant_not_found",
    )
    assert "tenant" in body["error"]["message"].lower()


@pytest.mark.parametrize("path", ROUTES)
def test_every_route_works_for_a_valid_tenant(api, headers_for, tenant_a, path):
    assert api.get(path, headers=headers_for(tenant_a)).status_code == 200


# --- isolation -------------------------------------------------------------------------
def test_same_order_number_returns_each_tenants_order(api, headers_for, tenant_a, tenant_b):
    a = api.get("/api/orders/ORD-1001", headers=headers_for(tenant_a)).json()
    b = api.get("/api/orders/ORD-1001", headers=headers_for(tenant_b)).json()
    assert (a["currency"], a["total_amount"], a["customer_code"]) == ("USD", "179.89", "CUS-1001")
    assert (b["currency"], b["total_amount"], b["customer_code"]) == ("EUR", "248.00", "CUS-1001")
    assert {i["sku"] for i in a["items"]} == {"SKU-1001", "SKU-1005"}
    assert {i["sku"] for i in b["items"]} == {"SKU-1001", "SKU-1004"}


def test_repeated_product_order_returns_two_distinct_lines(api, headers_for, tenant_a):
    items = api.get("/api/orders/ORD-1010", headers=headers_for(tenant_a)).json()["items"]
    assert sorted((i["sku"], i["quantity"], i["unit_price"]) for i in items) == [
        ("SKU-1005", 2, "19.96"),
        ("SKU-1005", 4, "24.95"),
    ]


def test_same_customer_code_returns_each_tenants_customer(api, headers_for, tenant_a, tenant_b):
    a = api.get("/api/customers/CUS-1001", headers=headers_for(tenant_a)).json()
    b = api.get("/api/customers/CUS-1001", headers=headers_for(tenant_b)).json()
    assert (a["name"], b["name"]) == ("Ava Thompson", "Emma Schneider")


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("/api/customers/CUS-1005", "customer_not_found"),
        ("/api/customers/CUS-1005/orders", "customer_not_found"),
        ("/api/customers/CUS-1006/invoices/unpaid", "customer_not_found"),
        ("/api/orders/ORD-1006", "order_not_found"),
        ("/api/invoices/INV-1005", "invoice_not_found"),
        ("/api/shipments/SHP-1005", "shipment_not_found"),
        ("/api/products/SKU-1005", "product_not_found"),
    ],
)
def test_tenant_a_only_references_are_invisible_to_tenant_b(
    api, headers_for, tenant_a, tenant_b, path, code
):
    assert api.get(path, headers=headers_for(tenant_a)).status_code == 200
    body_b = assert_error(api.get(path, headers=headers_for(tenant_b)), 404, code)
    # Indistinguishable from a reference that exists nowhere:
    missing = path.replace("1005", "9999").replace("1006", "9999")
    body_missing = assert_error(api.get(missing, headers=headers_for(tenant_b)), 404, code)
    assert body_b["error"]["message"].replace("1005", "X").replace("1006", "X") == body_missing[
        "error"
    ]["message"].replace("9999", "X")


def test_lists_only_contain_own_tenant_rows(api, headers_for, tenant_a, tenant_b):
    a = api.get("/api/customers", headers=headers_for(tenant_a)).json()["items"]
    b = api.get("/api/customers", headers=headers_for(tenant_b)).json()["items"]
    assert len(a) == 6 and len(b) == 4
    assert {c["email"].split("@")[1] for c in a} == {"example.com"}
    assert {c["email"].split("@")[1] for c in b} == {"example.org"}


def test_search_does_not_cross_tenants(api, headers_for, tenant_a, tenant_b):
    a = api.get("/api/customers?search=thompson", headers=headers_for(tenant_a)).json()
    b = api.get("/api/customers?search=thompson", headers=headers_for(tenant_b)).json()
    assert len(a["items"]) == 2 and b["items"] == []


# --- endpoint behaviour -----------------------------------------------------------------
def test_unpaid_invoices_endpoint(api, headers_for, tenant_a):
    body = api.get("/api/customers/CUS-1001/invoices/unpaid", headers=headers_for(tenant_a)).json()
    assert [(i["invoice_number"], i["is_overdue"]) for i in body["items"]] == [
        ("INV-1002", True),
        ("INV-1004", False),
    ]


def test_delayed_shipments_endpoint(api, headers_for, tenant_a, tenant_b):
    a = api.get("/api/shipments?status=delayed", headers=headers_for(tenant_a)).json()["items"]
    b = api.get("/api/shipments?status=delayed", headers=headers_for(tenant_b)).json()["items"]
    assert [(s["shipment_number"], s["delay_reason"] is not None) for s in a] == [
        ("SHP-1003", True)
    ]
    assert [(s["shipment_number"], s["delay_reason"]) for s in b] == [("SHP-1002", None)]


def test_demo_summary_endpoint(api, headers_for, tenant_a):
    body = api.get("/api/demo/summary", headers=headers_for(tenant_a)).json()
    assert body == {
        "tenant": {"name": "Northstar Commerce", "slug": "northstar-commerce"},
        "customers": 6,
        "products": 6,
        "orders": 10,
        "unpaid_invoices": 3,
        "overdue_invoices": 2,
        "delayed_shipments": 1,
    }


def test_money_serialised_as_exact_strings(api, headers_for, tenant_a):
    body = api.get("/api/products/SKU-1005", headers=headers_for(tenant_a)).json()
    assert body["unit_price"] == "24.95"


def test_responses_do_not_expose_internal_ids(api, headers_for, tenant_a):
    text = api.get("/api/orders/ORD-1001", headers=headers_for(tenant_a)).text
    assert str(tenant_a.tenant_id) not in text
    assert '"id"' not in text and "tenant_id" not in text


@pytest.mark.parametrize(
    "path", ["/api/shipments?status=lost", "/api/orders?status=", "/api/customers?limit=101"]
)
def test_invalid_query_params_use_error_envelope(api, headers_for, tenant_a, path):
    body = assert_error(api.get(path, headers=headers_for(tenant_a)), 422, "validation_error")
    assert "lost" not in body["error"]["message"]  # input is not echoed back


def test_unknown_route_uses_error_envelope(api, headers_for, tenant_a):
    assert_error(api.get("/api/nope", headers=headers_for(tenant_a)), 404, "route_not_found")


def test_read_only_api(api, headers_for, tenant_a):
    assert api.post("/api/customers", headers=headers_for(tenant_a)).status_code == 405
    assert api.delete("/api/orders/ORD-1001", headers=headers_for(tenant_a)).status_code == 405


def test_request_id_is_propagated(api, headers_for, tenant_a):
    resp = api.get("/api/products", headers={**headers_for(tenant_a), "X-Request-ID": "abc-123"})
    assert resp.headers["x-request-id"] == "abc-123"
    resp = api.get("/api/products", headers={**headers_for(tenant_a), "X-Request-ID": "bad id!"})
    assert resp.headers["x-request-id"] != "bad id!"
