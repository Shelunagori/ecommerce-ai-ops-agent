"""Tenant-context dependency and error envelope without a database."""

import logging
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_read_session
from app.main import create_app
from app.services.base import like_contains
from app.services.invoices import is_overdue
from tests.conftest import make_settings


@pytest.fixture
def fake_session():
    return MagicMock()


@pytest.fixture
def client(fake_session):
    app = create_app(make_settings())
    app.dependency_overrides[get_read_session] = lambda: fake_session

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internals: postgresql://u:hunter2@db/x")

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_missing_header(client, fake_session):
    resp = client.get("/api/customers")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tenant_context_missing"
    fake_session.scalar.assert_not_called()  # rejected before any DB work


def test_invalid_header(client):
    resp = client.get("/api/customers", headers={"X-Tenant-ID": "nope"})
    assert (resp.status_code, resp.json()["error"]["code"]) == (400, "tenant_context_invalid")


def test_unknown_tenant(client, fake_session):
    fake_session.scalar.return_value = None
    resp = client.get("/api/customers", headers={"X-Tenant-ID": str(uuid.uuid4())})
    assert (resp.status_code, resp.json()["error"]["code"]) == (404, "tenant_not_found")


def test_unhandled_errors_are_generic(client, caplog):
    with caplog.at_level(logging.ERROR):
        resp = client.get("/boom")
    assert resp.status_code == 500
    assert resp.json()["error"] == {
        "code": "internal_error",
        "message": "An unexpected error occurred.",
    }
    assert "hunter2" not in resp.text and "Traceback" not in resp.text


def test_logs_carry_request_and_tenant_ids(client, fake_session, caplog):
    tenant = str(uuid.uuid4())
    fake_session.scalar.return_value = None
    with caplog.at_level(logging.INFO, logger="app.request"):
        client.get("/api/customers", headers={"X-Tenant-ID": tenant, "X-Request-ID": "req-42"})
    [record] = [r for r in caplog.records if r.name == "app.request"]
    assert (record.request_id, record.tenant_id, record.status) == ("req-42", tenant, 404)


def test_malformed_tenant_header_is_not_logged(client, caplog):
    with caplog.at_level(logging.INFO, logger="app.request"):
        client.get("/api/customers", headers={"X-Tenant-ID": "<script>"})
    assert "<script>" not in caplog.text
    [record] = [r for r in caplog.records if r.name == "app.request"]
    assert not hasattr(record, "tenant_id")


def test_like_pattern_escapes_wildcards():
    assert like_contains("50%_off\\") == "%50\\%\\_off\\\\%"


def test_overdue_derivation():
    from datetime import UTC, datetime

    due = datetime(2026, 9, 1, tzinfo=UTC)
    after, before = datetime(2026, 9, 2, tzinfo=UTC), datetime(2026, 8, 31, tzinfo=UTC)
    assert is_overdue("pending", due, after) is True
    assert is_overdue("pending", due, before) is False
    assert is_overdue("paid", due, after) is False
    assert is_overdue("cancelled", due, after) is False
