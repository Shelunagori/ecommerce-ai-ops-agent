from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_settings


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/health",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )


def test_allowed_origin_gets_cors_headers(client):
    resp = _preflight(client, "http://localhost:3000")
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert resp.headers["access-control-allow-credentials"] == "true"


def test_unlisted_origin_is_rejected(client):
    resp = _preflight(client, "https://evil.example.com")
    assert "access-control-allow-origin" not in resp.headers


def test_wildcard_disables_credentials():
    app = create_app(make_settings(cors_origins=["*"]))
    with TestClient(app) as c:
        resp = _preflight(c, "https://anything.example.com")
    assert resp.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in resp.headers
