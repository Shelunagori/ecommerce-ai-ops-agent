"""Provider factory: the only place that knows about Ollama vs Gemini vs Cloudflare.

Construction performs no network I/O (no model validation, no discovery calls). All
provider- and model-specific parameters stay here. Chat inference only: the embedding
provider (``app.knowledge.embeddings``) is configured independently.
"""

import uuid
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.agent.llm.config import LLMConfig, ProviderName, cloudflare_base_url
from app.agent.llm.errors import LLMConfigurationError
from app.agent.llm.provider import (
    ChatModelProvider,
    FallbackProvider,
    LLMProvider,
    MessagePreparer,
    ProviderInfo,
    ResponseNormalizer,
    RetryPolicy,
    StructuredMethod,
)
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


# Cloudflare Workers AI: the OpenAI-compatible Chat Completions endpoint (tools, tool_calls
# with preserved ids, multi-turn tool round-trips). Called directly from the backend.
CLOUDFLARE_MAX_OUTPUT_TOKENS = 1024


def _build_cloudflare(
    config: LLMConfig, *, http_client: httpx.Client | None = None
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    assert config.cloudflare_account_id is not None  # guaranteed by LLMConfig
    assert config.cloudflare_api_token is not None
    return ChatOpenAI(
        model=config.model,
        base_url=cloudflare_base_url(config.cloudflare_account_id),
        api_key=config.cloudflare_api_token,  # explicit; never read implicitly from env
        timeout=config.timeout_seconds,
        max_retries=0,  # retries are owned by ChatModelProvider's RetryPolicy
        temperature=0,
        # Workers AI's native field (its default output budget is small); sent verbatim.
        extra_body={"max_tokens": CLOUDFLARE_MAX_OUTPUT_TOKENS},
        stream_usage=False,
        http_socket_options=(),  # keep httpx defaults (proxy env honoured)
        http_client=http_client,  # tests inject a mock transport; None = SDK default
    )


def openai_compatible_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """History for an OpenAI-compatible endpoint: plain text + tool calls only.

    A thread may hold turns another provider produced (e.g. Gemini content blocks, thought
    signatures). Those provider-specific parts are dropped for this request only; the
    checkpointed history is unchanged and tool-call ids are preserved exactly."""
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, AIMessage):
            out.append(AIMessage(content=m.text, tool_calls=list(m.tool_calls), id=m.id))
        elif isinstance(m, ToolMessage) and not isinstance(m.content, str):
            out.append(
                ToolMessage(
                    content=m.text, tool_call_id=m.tool_call_id, name=m.name, status=m.status
                )
            )
        else:
            out.append(m)
    return out


CORRELATION_ID_PREFIX = "cf_call_"


def _usable_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def normalize_tool_call_ids(ai: AIMessage) -> tuple[AIMessage, dict[str, Any]]:
    """Workers AI (OpenAI-compatible) can return a STRUCTURED tool call - known name, JSON
    arguments - whose ``id`` is missing, null or empty. The graph requires an id to link the
    call to its ToolMessage, so exactly those calls get a server-side correlation id
    (``cf_call_<uuid4 hex>``). The id carries no authority; it is only a message link.

    Never changed: a provider-supplied non-empty id (even a duplicate - ambiguous, so the
    graph refuses it), tool names, arguments, ``invalid_tool_calls`` (unparseable arguments
    stay unparseable) and the message text. The provider's message object is not mutated."""
    calls = list(ai.tool_calls)
    if not calls:
        return ai, {}
    fixed = []
    generated: dict[int, str] = {}
    for i, call in enumerate(calls):
        if _usable_id(call.get("id")):
            fixed.append(call)
        else:
            generated[i] = f"{CORRELATION_ID_PREFIX}{uuid.uuid4().hex}"
            fixed.append({**call, "id": generated[i]})
    facts: dict[str, Any] = {"tool_calls": len(calls)}
    if not generated:
        return ai, facts
    update: dict[str, Any] = {"tool_calls": fixed}
    raw = ai.additional_kwargs.get("tool_calls")
    if isinstance(raw, list) and len(raw) == len(calls):
        # keep the raw OpenAI-format copy consistent with the parsed calls
        update["additional_kwargs"] = {
            **ai.additional_kwargs,
            "tool_calls": [
                {**r, "id": generated[i]} if i in generated and isinstance(r, dict) else r
                for i, r in enumerate(raw)
            ],
        }
    facts["tool_call_id_normalized"] = len(generated)
    return ai.model_copy(update=update), facts


_BUILDERS: dict[ProviderName, Callable[[LLMConfig], BaseChatModel]] = {
    "ollama": _build_ollama,
    "gemini": _build_gemini,
    "cloudflare": _build_cloudflare,
}
# Workers AI JSON mode covers only some models, so structured output uses one forced tool
# call there; results are re-validated with the full Pydantic schema either way.
_STRUCTURED: dict[ProviderName, StructuredMethod] = {"cloudflare": "tool_call"}
_PREPARE: dict[ProviderName, MessagePreparer] = {"cloudflare": openai_compatible_messages}
_NORMALIZE: dict[ProviderName, ResponseNormalizer] = {"cloudflare": normalize_tool_call_ids}


def build_provider(config: LLMConfig, *, http_client: httpx.Client | None = None) -> LLMProvider:
    """``http_client``: test hook for the OpenAI-compatible (Cloudflare) client only."""
    builder = _BUILDERS.get(config.provider)
    if builder is None:  # unreachable via LLMConfig, kept as a hard guard
        raise LLMConfigurationError("Unsupported LLM provider.")
    try:
        if config.provider == "cloudflare":
            chat_model = _build_cloudflare(config, http_client=http_client)
        else:
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
        structured_method=_STRUCTURED.get(config.provider, "json_schema"),
        prepare_messages=_PREPARE.get(config.provider),
        normalize_response=_NORMALIZE.get(config.provider),
    )


def get_llm_provider(
    settings: Settings | None = None, *, provider: str | None = None, model: str | None = None
) -> LLMProvider:
    """Build the configured provider. ``provider``/``model`` override settings (CLI use; an
    explicit override never adds a fallback). With ``LLM_FALLBACK_PROVIDER`` set, the result
    is a ``FallbackProvider`` (per-model-call fallback on availability errors only)."""
    settings = settings or get_settings()
    config = LLMConfig.from_settings(settings, provider=provider, model=model)
    primary = build_provider(config)
    fallback_name = settings.llm_fallback_provider
    if provider is not None or model is not None or fallback_name is None:
        return primary
    if fallback_name == config.provider:
        raise LLMConfigurationError("LLM_FALLBACK_PROVIDER must differ from LLM_PROVIDER.")
    fallback = build_provider(LLMConfig.from_settings(settings, provider=fallback_name))
    return FallbackProvider(primary, fallback)
