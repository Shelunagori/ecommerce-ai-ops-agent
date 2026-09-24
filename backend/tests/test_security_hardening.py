"""Phase 12 security gaps: agent rate limiting, API security headers, no public API docs in
production. No DB, no network."""

import pytest
from fastapi.testclient import TestClient

from app.api.ratelimit import RateLimiter
from app.main import create_app
from tests.conftest import make_settings
from tests.test_production import prod


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_sliding_window_per_key():
    clock = Clock()
    limiter = RateLimiter(2, window_seconds=60, clock=clock)
    assert limiter.hit("a") is None and limiter.hit("a") is None
    retry = limiter.hit("a")
    assert retry is not None and 0 < retry <= 60
    assert limiter.hit("b") is None  # other principals are unaffected
    clock.now += 61
    assert limiter.hit("a") is None  # window slid


def test_zero_disables_and_keys_are_bounded():
    assert all(RateLimiter(0).hit("a") is None for _ in range(100))
    clock = Clock()
    limiter = RateLimiter(1, window_seconds=1, clock=clock, max_keys=10)
    for i in range(50):
        limiter.hit(f"user-{i}")
        clock.now += 2  # every previous window expired
    assert limiter.tracked_keys() <= 10


@pytest.mark.parametrize("path", ["/health", "/api/customers"])
def test_security_headers_on_every_response(path):
    r = TestClient(create_app(make_settings()), raise_server_exceptions=False).get(path)
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["x-frame-options"] == "DENY"
    if path.startswith("/api"):
        assert r.headers["cache-control"] == "no-store"  # tenant data is never cached


def test_api_docs_are_not_published_in_production():
    client = TestClient(create_app(prod()))
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
    assert TestClient(create_app(make_settings())).get("/openapi.json").status_code == 200


def test_unhandled_errors_keep_request_id_and_headers_without_leaking():
    app = create_app(make_settings())

    @app.get("/api/boom")
    def boom():
        raise RuntimeError("postgresql://admin:hunter2@db.internal/prod")

    r = TestClient(app, raise_server_exceptions=False).get(
        "/api/boom", headers={"X-Request-ID": "req-boom-1"}
    )
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
    assert r.headers["x-request-id"] == "req-boom-1" == r.json()["request_id"]
    assert r.headers["x-content-type-options"] == "nosniff" and "hunter2" not in r.text


def test_request_log_never_contains_the_query_string(client, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="app.request"):
        client.get("/health?access_token=SECRET-QS-123&email=someone@example.com")
    [record] = [r for r in caplog.records if r.name == "app.request"]
    assert record.path == "/health"
    everything = " ".join(str(v) for v in vars(record).values())  # message AND extra fields
    assert "SECRET-QS-123" not in everything and "someone@example.com" not in everything
