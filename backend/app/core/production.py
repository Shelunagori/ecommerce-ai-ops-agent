"""Production configuration validation (Phase 11).

``configuration_problems(settings)`` returns stable problem CODES (setting names, never
values) for a configuration that must not serve production traffic. The server refuses to
start with any problem when ``APP_ENV=production`` (see ``app.main`` lifespan), the readiness
endpoint reports ``config`` as failed, and ``scripts/check_env.py`` runs the same check as a
pre-deploy step. Nothing here makes a network call or reads a secret's value beyond
"is it set".
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from pydantic import SecretStr

from app.core.config import Settings


def _https(url: str | None) -> bool:
    parsed = urlparse(url or "")
    return parsed.scheme == "https" and bool(parsed.netloc)


def _set(secret: SecretStr | None) -> bool:
    return secret is not None and bool(secret.get_secret_value().strip())


def configuration_problems(settings: Settings) -> list[str]:
    """Empty unless ``app_env == "production"``; otherwise every unmet requirement."""
    if settings.app_env != "production":
        return []
    problems: list[str] = []
    if not settings.database_url:
        problems.append("DATABASE_URL:missing")
    if settings.debug:
        problems.append("DEBUG:must_be_false")
    # Trusted identity only: the demo X-Tenant-ID header is never accepted in production.
    if settings.auth_mode != "supabase":
        problems.append("AUTH_MODE:must_be_supabase")
    if not _https(settings.supabase_url):
        problems.append("SUPABASE_URL:missing_or_not_https")
    # Browsers: explicit HTTPS origins only (no wildcard, no plain HTTP).
    if not settings.cors_origins:
        problems.append("CORS_ORIGINS:missing")
    elif any(o == "*" or not _https(o) for o in settings.cors_origins):
        problems.append("CORS_ORIGINS:must_be_explicit_https")
    # Hosted inference: there is no local Ollama server next to a hosted API. The chat
    # provider (+ optional fallback) and the embedding provider are configured independently.
    hosted_chat = ("gemini", "cloudflare")
    if settings.llm_provider not in hosted_chat:
        problems.append("LLM_PROVIDER:must_be_hosted")
    fallback = settings.llm_fallback_provider
    if fallback is not None and fallback not in hosted_chat:
        problems.append("LLM_FALLBACK_PROVIDER:must_be_hosted")
    if fallback is not None and fallback == settings.llm_provider:
        problems.append("LLM_FALLBACK_PROVIDER:must_differ_from_primary")
    if settings.embedding_provider != "gemini":
        problems.append("EMBEDDING_PROVIDER:must_be_hosted")
    chat_providers = {settings.llm_provider, fallback}
    if (
        "gemini" in chat_providers or settings.embedding_provider == "gemini"
    ) and settings.gemini_api_key is None:
        problems.append("GEMINI_API_KEY:missing")
    if "cloudflare" in chat_providers:
        if not re.fullmatch(r"[0-9a-f]{32}", (settings.cloudflare_account_id or "").strip()):
            problems.append("CLOUDFLARE_ACCOUNT_ID:missing_or_invalid")
        if not _set(settings.cloudflare_api_token):
            problems.append("CLOUDFLARE_API_TOKEN:missing")
        if not (settings.cloudflare_model or "").strip():
            problems.append("CLOUDFLARE_MODEL:missing")
    if settings.langsmith_tracing and settings.langsmith_api_key is None:
        problems.append("LANGSMITH_API_KEY:missing_while_tracing_enabled")
    return problems


class ProductionConfigurationError(RuntimeError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("invalid production configuration: " + ", ".join(problems))


def require_valid_production_configuration(settings: Settings) -> None:
    problems = configuration_problems(settings)
    if problems:
        raise ProductionConfigurationError(problems)
