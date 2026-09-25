"""Validated LLM configuration derived from application settings.

This is the configuration boundary: unknown providers, a missing Gemini key / Cloudflare
account or token, or malformed values fail here with a clear ``LLMConfigurationError``
whose message never includes a secret. Nothing here performs network I/O.
"""

import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

from pydantic import SecretStr

from app.agent.llm.errors import LLMConfigurationError
from app.core.config import Settings

ProviderName = Literal["ollama", "gemini", "cloudflare"]
SUPPORTED_PROVIDERS: tuple[ProviderName, ...] = ("ollama", "gemini", "cloudflare")
MAX_RETRIES_CAP = 2

# Model ids: "qwen3:4b-instruct", "gemini-3.8-flash", "@cf/meta/llama-4-scout-17b-16e-instruct"
_SAFE_NAME = re.compile(r"^@?[A-Za-z0-9._:/-]{1,100}$")
# A Cloudflare account id is 32 lowercase hex characters (it becomes part of the URL path).
_CF_ACCOUNT = re.compile(r"^[0-9a-f]{32}$")
CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4/accounts"


def cloudflare_base_url(account_id: str) -> str:
    """OpenAI-compatible Workers AI base URL (``/chat/completions`` is appended by the SDK)."""
    if not _CF_ACCOUNT.fullmatch(account_id):
        raise LLMConfigurationError("CLOUDFLARE_ACCOUNT_ID must be a 32-character hex id.")
    return f"{CLOUDFLARE_API_BASE}/{account_id}/ai/v1"


@dataclass(frozen=True)
class LLMConfig:
    provider: ProviderName
    model: str
    timeout_seconds: float
    max_retries: int
    ollama_base_url: str
    gemini_api_key: SecretStr | None = field(default=None, repr=False)
    cloudflare_account_id: str | None = field(default=None, repr=False)
    cloudflare_api_token: SecretStr | None = field(default=None, repr=False)

    @classmethod
    def from_settings(
        cls, settings: Settings, *, provider: str | None = None, model: str | None = None
    ) -> "LLMConfig":
        name = (provider or settings.llm_provider or "").strip().lower()
        if name not in SUPPORTED_PROVIDERS:
            shown = f" '{name}'" if re.fullmatch(r"[a-z0-9_-]{1,32}", name) else ""
            raise LLMConfigurationError(
                f"Unsupported LLM provider{shown}. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
            )

        default_model = {
            "ollama": settings.ollama_model,
            "gemini": settings.gemini_model,
            "cloudflare": settings.cloudflare_model,
        }[name]
        chosen = (model or default_model or "").strip()
        if not _SAFE_NAME.fullmatch(chosen):
            raise LLMConfigurationError(f"A valid model name is required for provider '{name}'.")

        timeout = float(settings.llm_timeout_seconds)
        if not 1 <= timeout <= 300:
            raise LLMConfigurationError("LLM_TIMEOUT_SECONDS must be between 1 and 300.")
        retries = int(settings.llm_max_retries)
        if not 0 <= retries <= MAX_RETRIES_CAP:
            raise LLMConfigurationError(f"LLM_MAX_RETRIES must be between 0 and {MAX_RETRIES_CAP}.")

        base_url = settings.ollama_base_url.strip().rstrip("/")
        key = settings.gemini_api_key
        if name == "ollama":
            parsed = urlparse(base_url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise LLMConfigurationError("OLLAMA_BASE_URL must be an http(s) URL.")
        if name == "gemini" and (key is None or not key.get_secret_value().strip()):
            raise LLMConfigurationError("GEMINI_API_KEY is required when the provider is gemini.")
        account = (settings.cloudflare_account_id or "").strip().lower()
        token = settings.cloudflare_api_token
        if name == "cloudflare":
            if not _CF_ACCOUNT.fullmatch(account):
                raise LLMConfigurationError(
                    "CLOUDFLARE_ACCOUNT_ID (32-character hex) is required for provider cloudflare."
                )
            if token is None or not token.get_secret_value().strip():
                raise LLMConfigurationError(
                    "CLOUDFLARE_API_TOKEN is required when the provider is cloudflare."
                )

        return cls(
            provider=name,  # type: ignore[arg-type]
            model=chosen,
            timeout_seconds=timeout,
            max_retries=retries,
            ollama_base_url=base_url,
            gemini_api_key=key if name == "gemini" else None,
            cloudflare_account_id=account if name == "cloudflare" else None,
            cloudflare_api_token=token if name == "cloudflare" else None,
        )
