"""Deployment readiness (Phase 11): production configuration validation, fail-fast startup,
the readiness endpoint contract and the pre-deploy ``check_env`` script. No DB, no network."""

import pytest
from fastapi.testclient import TestClient

from app.api.health import get_readiness_check
from app.core.production import (
    ProductionConfigurationError,
    configuration_problems,
)
from app.main import create_app
from tests.conftest import make_settings

SECRET = "AIza-NEVER-PRINT-ME-000"
VALID = {
    "app_env": "production",
    "database_url": "postgresql://u:p@pooler.example.com:5432/postgres?sslmode=require",
    "auth_mode": "supabase",
    "supabase_url": "https://demo-project.supabase.co",
    "cors_origins": ["https://commerceops.example.app"],
    "llm_provider": "gemini",
    "embedding_provider": "gemini",
    "gemini_api_key": SECRET,
}


def prod(**over):
    return make_settings(**{**VALID, **over})


def test_valid_production_configuration():
    assert configuration_problems(prod()) == []


def test_non_production_is_never_blocked():
    assert configuration_problems(make_settings(app_env="development")) == []


@pytest.mark.parametrize(
    ("over", "problem"),
    [
        ({"database_url": None}, "DATABASE_URL:missing"),
        ({"debug": True}, "DEBUG:must_be_false"),
        ({"auth_mode": "demo"}, "AUTH_MODE:must_be_supabase"),
        ({"supabase_url": None}, "SUPABASE_URL:missing_or_not_https"),
        ({"supabase_url": "http://demo.supabase.co"}, "SUPABASE_URL:missing_or_not_https"),
        ({"cors_origins": []}, "CORS_ORIGINS:missing"),
        ({"cors_origins": ["*"]}, "CORS_ORIGINS:must_be_explicit_https"),
        ({"cors_origins": ["http://localhost:3000"]}, "CORS_ORIGINS:must_be_explicit_https"),
        ({"llm_provider": "ollama"}, "LLM_PROVIDER:must_be_hosted"),
        ({"embedding_provider": "ollama"}, "EMBEDDING_PROVIDER:must_be_hosted"),
        ({"gemini_api_key": None}, "GEMINI_API_KEY:missing"),
        ({"langsmith_tracing": True}, "LANGSMITH_API_KEY:missing_while_tracing_enabled"),
    ],
)
def test_each_requirement(over, problem):
    problems = configuration_problems(prod(**over))
    assert problem in problems and len(problems) == 1


def test_server_refuses_to_start_with_an_unsafe_production_config():
    app = create_app(prod(auth_mode="demo", gemini_api_key=None, llm_provider="ollama"))
    with pytest.raises(ProductionConfigurationError) as got, TestClient(app):
        pass  # pragma: no cover - startup fails first
    assert "AUTH_MODE:must_be_supabase" in str(got.value) and SECRET not in str(got.value)


def test_server_starts_with_a_valid_production_config():
    app = create_app(prod())
    app.dependency_overrides[get_readiness_check] = lambda: lambda _s: {"config": "ok"}
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize(
    ("checks", "code", "status"),
    [
        ({"config": "ok", "database": "ok", "migrations": "ok", "checkpoints": "ok"}, 200, "ready"),
        (
            {"config": "ok", "database": "ok", "migrations": "not_at_head", "checkpoints": "ok"},
            503,
            "not_ready",
        ),
    ],
)
def test_readiness_contract(checks, code, status):
    app = create_app(make_settings())
    app.dependency_overrides[get_readiness_check] = lambda: lambda _s: checks
    r = TestClient(app).get("/health/ready")
    assert (r.status_code, r.json()) == (code, {"status": status, "checks": checks})


@pytest.fixture
def no_database(monkeypatch):
    """Hermetic "no DATABASE_URL": the probe uses the process-wide engine, which reads the
    environment / backend/.env (a developer machine usually has one), so isolate it here."""
    import app.db.session as db

    monkeypatch.setattr(db, "get_settings", lambda: make_settings(database_url=None))
    db.get_engine.cache_clear()
    db._session_factory.cache_clear()
    yield
    db.get_engine.cache_clear()
    db._session_factory.cache_clear()


def test_readiness_without_a_database_is_not_ready(no_database):
    r = TestClient(create_app(make_settings())).get("/health/ready")
    assert r.status_code == 503
    assert r.json()["checks"] == {
        "config": "ok",
        "database": "not_configured",
        "migrations": "unknown",
        "checkpoints": "unknown",
    }


def test_check_env_script(monkeypatch, capsys):
    from scripts import check_env

    for key in list(VALID):
        monkeypatch.delenv(key.upper(), raising=False)
    monkeypatch.chdir("/")  # no .env file
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUTH_MODE", "demo")
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    assert check_env.main([]) == 1
    err = capsys.readouterr().err
    assert "AUTH_MODE:must_be_supabase" in err and "DATABASE_URL:missing" in err
    assert SECRET not in err
    for key, value in VALID.items():
        if key == "cors_origins":
            value = ",".join(value)
        monkeypatch.setenv(key.upper(), str(value))
    assert check_env.main([]) == 0
    assert SECRET not in capsys.readouterr().out
    monkeypatch.setenv("APP_ENV", "nonsense")
    assert check_env.main([]) == 1
    assert "APP_ENV" in capsys.readouterr().err
