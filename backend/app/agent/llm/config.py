"""Validated LLM configuration derived from application settings.

This is the configuration boundary: unknown providers, a missing Gemini key or malformed
values fail here with a clear ``LLMConfigurationError`` whose message never includes a
secret. Nothing here performs network I/O.
"""

import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

from pydantic import SecretStr

from app.agent.llm.errors import LLMConfigurationError
from app.core.config import Settings

ProviderName = Literal["ollama", "gemini"]
SUPPORTED_PROVIDERS: tuple[ProviderName, ...] = ("ollama", "gemini")
MAX_RETRIES_CAP = 2

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._:/-]{1,100}$")


@dataclass(frozen=True)
class LLMConfig:
    provider: ProviderName
    model: str
    timeout_seconds: float
    max_retries: int
    ollama_base_url: str
    gemini_api_key: SecretStr | None = field(default=None, repr=False)

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

        default_model = settings.ollama_model if name == "ollama" else settings.gemini_model
        chosen = (model or default_model or "").strip()
        if not _SAFE_NAME.fullmatch(chosen):
            raise LLMConfigurationError(f"A valid model name is required for provider '{name}'.")

        timeout = float(settings.llm_timeout_seconds)
        if not 0 < timeout <= 300:
            raise LLMConfigurationError("LLM_TIMEOUT_SECONDS must be between 0 and 300.")
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

        return cls(
            provider=name,  # type: ignore[arg-type]
            model=chosen,
            timeout_seconds=timeout,
            max_retries=retries,
            ollama_base_url=base_url,
            gemini_api_key=key if name == "gemini" else None,
        )
