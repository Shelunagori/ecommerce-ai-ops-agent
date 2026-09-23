"""Environment-based application settings."""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "test", "staging", "production"] = "development"
    app_name: str = "commerceops-api"
    debug: bool = False

    # Optional so the process can start (and /health answer) without a DB.
    database_url: str | None = None

    # Comma-separated in the environment: "http://localhost:3000,https://x.vercel.app"
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # LLM provider layer (app/agent/llm). Provider-specific values are only read by the
    # provider factory. No network call happens until a model is actually invoked.
    llm_provider: Literal["ollama", "gemini"] = "ollama"
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    llm_max_retries: int = Field(default=1, ge=0, le=2)  # transient failures only
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [o.strip().rstrip("/") for o in value.split(",") if o.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
