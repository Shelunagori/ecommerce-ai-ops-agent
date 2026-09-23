import logging

import pytest

from app.db.session import DatabaseNotConfiguredError
from app.schemas.health import DatabaseHealthResponse, HealthResponse


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "commerceops-api"}


def test_health_matches_schema(client):
    HealthResponse.model_validate(client.get("/health").json())


def test_health_db_reachable(client, override_db_check):
    override_db_check(lambda: None)
    resp = client.get("/health/db")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "reachable"}
    DatabaseHealthResponse.model_validate(resp.json())


def test_health_db_unreachable_returns_503_without_leaking(client, override_db_check, caplog):
    secret_url = "postgresql://admin:SuperSecret123@db.internal:5432/prod"

    def failing() -> None:
        raise ConnectionError(f"could not connect to {secret_url}")

    override_db_check(failing)
    with caplog.at_level(logging.WARNING):
        resp = client.get("/health/db")

    assert resp.status_code == 503
    assert resp.json() == {"status": "error", "database": "unreachable"}
    assert "SuperSecret123" not in resp.text
    assert "SuperSecret123" not in caplog.text
    assert "db.internal" not in caplog.text


def test_health_db_not_configured_returns_503(client, override_db_check):
    def not_configured() -> None:
        raise DatabaseNotConfiguredError("DATABASE_URL is not set")

    override_db_check(not_configured)
    resp = client.get("/health/db")
    assert resp.status_code == 503
    assert resp.json() == {"status": "error", "database": "not_configured"}


@pytest.mark.parametrize("path", ["/health", "/health/db"])
def test_openapi_documents_health_routes(client, path):
    assert path in client.get("/openapi.json").json()["paths"]
