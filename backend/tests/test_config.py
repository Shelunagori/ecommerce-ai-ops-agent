from app.core.config import Settings


def test_cors_origins_parsed_from_comma_separated_env(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, https://app.vercel.app/ ,")
    s = Settings(_env_file=None)
    assert s.cors_origins == ["http://localhost:3000", "https://app.vercel.app"]


def test_defaults_do_not_require_any_env(monkeypatch):
    for var in ("DATABASE_URL", "CORS_ORIGINS", "GEMINI_API_KEY", "APP_ENV"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.database_url is None
    assert s.cors_origins == []
    assert s.gemini_api_key is None


def test_gemini_key_is_masked_in_repr(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder-not-a-real-key")
    s = Settings(_env_file=None)
    assert "placeholder-not-a-real-key" not in repr(s)
    assert s.gemini_api_key is not None
    assert s.gemini_api_key.get_secret_value() == "placeholder-not-a-real-key"
