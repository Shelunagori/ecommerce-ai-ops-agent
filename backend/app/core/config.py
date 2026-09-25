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

    # Embeddings (Step 8) - separate from the chat-model settings above. The Ollama server
    # is the same ``ollama_base_url``; the model is a dedicated embedding model.
    embedding_provider: Literal["ollama", "gemini"] = "ollama"
    ollama_embedding_model: str = "nomic-embed-text-v2-moe"
    # Hosted embeddings (Phase 8): Gemini API, key = GEMINI_API_KEY. A separate profile.
    gemini_embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: int = Field(default=768, ge=1, le=16000)
    embedding_timeout_seconds: float = Field(default=60.0, ge=1, le=300)
    embedding_batch_size: int = Field(default=16, ge=1, le=64)
    embedding_max_retries: int = Field(default=1, ge=0, le=2)  # transient failures only

    # Approval-gated actions (Step 10). Trusted configuration only.
    action_approval_ttl_seconds: int = Field(default=900, ge=30, le=86_400)
    action_execution_claim_timeout_seconds: int = Field(default=120, ge=5, le=3600)
    # Upper bound for one synthetic store credit (Decimal string; never a float).
    store_credit_max_amount: str = Field(default="100.00", pattern=r"^\d{1,7}(\.\d{1,2})?$")

    # Abuse protection (Phase 12): chat messages per (user, tenant) per minute; 0 disables.
    agent_rate_limit_per_minute: int = Field(default=20, ge=0, le=1000)

    # Public "Try Live Demo" (Supabase Anonymous Sign-Ins). OFF by default. A VERIFIED
    # anonymous JWT (``is_anonymous: true``) gets read-only ``member`` access to exactly this
    # one synthetic tenant; no membership rows are created. Permanent users are unaffected.
    public_demo_enabled: bool = False
    public_demo_tenant_slug: str = Field(default="bluepeak-retail", min_length=1, max_length=64)
    # Stricter budgets for anonymous visitors: per visitor, and shared by ALL visitors (new
    # anonymous identities are cheap to create, so a per-identity limit alone is not enough).
    public_demo_rate_limit_per_minute: int = Field(default=5, ge=0, le=1000)
    public_demo_global_rate_limit_per_minute: int = Field(default=60, ge=0, le=10_000)

    # Authentication / tenant boundary (Phase 7).
    #   demo     - X-Tenant-ID header is trusted (LOCAL DEMO ONLY; refused in production)
    #   supabase - Supabase Auth JWT (JWKS: RS256/ES256) + server-side tenant_memberships
    auth_mode: Literal["demo", "supabase"] = "demo"
    supabase_url: str | None = None  # https://<project-ref>.supabase.co
    supabase_jwt_audience: str = "authenticated"
    # Legacy shared-secret (HS256) projects only; asymmetric JWKS keys are preferred.
    supabase_jwt_secret: SecretStr | None = None
    auth_jwks_cache_seconds: int = Field(default=300, ge=30, le=86_400)
    demo_user_subject: str = "demo-user"

    # Optional LangSmith tracing (Phase 5). OFF by default; nothing depends on it. When on,
    # prompts and model outputs are sent to LangSmith - synthetic data only.
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "commerceops-ai"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [o.strip().rstrip("/") for o in value.split(",") if o.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
