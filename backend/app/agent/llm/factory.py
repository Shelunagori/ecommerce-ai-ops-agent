"""Provider factory: the only place that knows about Ollama vs Gemini.

Construction performs no network I/O (no model validation, no discovery calls). All
provider- and model-specific parameters stay here.
"""

from collections.abc import Callable

import httpx
from langchain_core.language_models import BaseChatModel

from app.agent.llm.config import LLMConfig, ProviderName
from app.agent.llm.errors import LLMConfigurationError
from app.agent.llm.provider import ChatModelProvider, LLMProvider, ProviderInfo, RetryPolicy
from app.core.config import Settings, get_settings

# Output is a small JSON object; cap generation for local models to bound latency.
OLLAMA_NUM_PREDICT = 512


def _build_ollama(config: LLMConfig) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.model,
        base_url=config.ollama_base_url,
        temperature=0,
        num_predict=OLLAMA_NUM_PREDICT,
        validate_model_on_init=False,  # no network during construction
        # No reasoning/think setting is forced: the default (qwen3:4b-instruct) is a
        # non-thinking model, and Ollama may reject `think` for models that lack it.
        client_kwargs={"timeout": httpx.Timeout(config.timeout_seconds)},
    )


def _gemini_thinking_kwargs(model: str) -> dict[str, str]:
    # Gemini 3.x uses thinking levels; "low" is the lowest level supported by 3.8 Flash
    # ("minimal" is rejected). For other model families we set nothing and use defaults.
    return {"thinking_level": "low"} if model.startswith("gemini-3") else {}


def _build_gemini(config: LLMConfig) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    assert config.gemini_api_key is not None  # guaranteed by LLMConfig
    return ChatGoogleGenerativeAI(
        model=config.model,
        api_key=config.gemini_api_key,  # passed explicitly; never read implicitly from env
        timeout=config.timeout_seconds,
        # The Google SDK treats 0 as "SDK default retries"; 1 means a single attempt.
        # Retries are owned by ChatModelProvider's RetryPolicy instead.
        max_retries=1,
        **_gemini_thinking_kwargs(config.model),
    )


_BUILDERS: dict[ProviderName, Callable[[LLMConfig], BaseChatModel]] = {
    "ollama": _build_ollama,
    "gemini": _build_gemini,
}


def build_provider(config: LLMConfig) -> LLMProvider:
    builder = _BUILDERS.get(config.provider)
    if builder is None:  # unreachable via LLMConfig, kept as a hard guard
        raise LLMConfigurationError("Unsupported LLM provider.")
    try:
        chat_model = builder(config)
    except LLMConfigurationError:
        raise
    except Exception:  # noqa: BLE001 - never surface constructor text (could echo config)
        raise LLMConfigurationError(
            f"The '{config.provider}' model client could not be configured."
        ) from None
    return ChatModelProvider(
        chat_model,
        ProviderInfo(provider=config.provider, model=config.model),
        RetryPolicy(max_retries=config.max_retries),
    )


def get_llm_provider(
    settings: Settings | None = None, *, provider: str | None = None, model: str | None = None
) -> LLMProvider:
    """Build the configured provider. ``provider``/``model`` override settings (CLI use)."""
    config = LLMConfig.from_settings(settings or get_settings(), provider=provider, model=model)
    return build_provider(config)
