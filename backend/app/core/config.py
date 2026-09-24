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
    llm_timeout_seconds: float = Field(default=60.0, ge=1, le=300)  # seconds
    llm_max_retries: int = Field(default=1, ge=0, le=2)  # transient failures only
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"

    # Commerce assistant loop bounds (trusted configuration only; never model input).
    assistant_max_model_rounds: int = Field(default=5, ge=1, le=10)
    assistant_max_tool_calls: int = Field(default=8, ge=1, le=20)
    assistant_max_tool_calls_per_turn: int = Field(default=4, ge=1, le=8)

    # Knowledge / policy chunking (Step 7). Trusted configuration; changing it changes the
    # recorded chunking_hash, and ingestion then refuses to replace existing chunks.
    knowledge_chunk_max_chars: int = Field(default=1200, ge=200, le=4000)
    knowledge_chunk_overlap_chars: int = Field(default=0, ge=0, le=300)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [o.strip().rstrip("/") for o in value.split(",") if o.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
