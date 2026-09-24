"""Provider factory and configuration boundary (offline)."""

import re
from pathlib import Path

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from app.agent.llm import LLMConfigurationError, get_llm_provider
from app.agent.llm.config import LLMConfig
from app.agent.llm.intent import IntentAnalysis
from app.agent.llm.provider import provider_json_schema
from tests.llm.conftest import FAKE_KEY, settings


def test_ollama_provider(no_network):
    p = get_llm_provider(settings(llm_provider="ollama"))
    assert (p.info.provider, p.info.model) == ("ollama", "qwen3:4b-instruct")
    chat = p._chat_model
    assert isinstance(chat, ChatOllama)
    assert chat.base_url == "http://localhost:11434" and chat.temperature == 0
    assert chat.validate_model_on_init is False
    assert chat.client_kwargs["timeout"].read == 60.0
    assert no_network == []


def test_gemini_provider(no_network):
    p = get_llm_provider(settings(llm_provider="gemini", gemini_api_key=FAKE_KEY))
    assert (p.info.provider, p.info.model) == ("gemini", "gemini-3.8-flash")
    chat = p._chat_model
    assert isinstance(chat, ChatGoogleGenerativeAI)
    assert chat.google_api_key.get_secret_value() == FAKE_KEY
    assert chat.timeout == 60.0
    assert chat.max_retries == 1  # Google SDK: 1 == single attempt; our RetryPolicy retries
    assert chat.reasoning_effort == "low"  # thinking_level for Gemini 3.x
    assert chat.temperature is None  # left to the model default for Gemini 3
    assert no_network == []


def test_thinking_level_only_set_for_gemini_3_models():
    chat = get_llm_provider(
        settings(llm_provider="gemini", gemini_api_key=FAKE_KEY), model="gemini-2.5-flash"
    )._chat_model
    assert chat.reasoning_effort is None and chat.thinking_budget is None


@pytest.mark.parametrize("model", ["llama3.2:3b", "qwen3:4b", "qwen2.5:7b"])
def test_other_ollama_models_can_still_be_selected_explicitly(model):
    assert get_llm_provider(settings(ollama_model=model)).info.model == model
    assert get_llm_provider(settings(), model=model).info.model == model


def test_overrides_and_timeouts_flow_from_settings():
    p = get_llm_provider(settings(llm_timeout_seconds=12, llm_max_retries=2), model="qwen2.5:1.5b")
    assert p.info.model == "qwen2.5:1.5b"
    assert p._chat_model.client_kwargs["timeout"].read == 12
    assert p._retry.max_retries == 2


def test_construction_and_structured_binding_perform_no_network_io(no_network):
    for provider in ("ollama", "gemini"):
        p = get_llm_provider(settings(gemini_api_key=FAKE_KEY), provider=provider)
        p._chat_model.with_structured_output(
            provider_json_schema(IntentAnalysis), method="json_schema", include_raw=True
        )
    assert no_network == []


@pytest.mark.parametrize("name", ["openai", "GEMINIX", "", "  "])
def test_unknown_provider_fails_at_factory_boundary(name):
    with pytest.raises(LLMConfigurationError) as exc:
        get_llm_provider(settings(), provider=name or " ")
    assert "Supported: ollama, gemini" in exc.value.message
    assert exc.value.code == "llm_not_configured"


def test_unsafe_provider_value_is_not_echoed():
    with pytest.raises(LLMConfigurationError) as exc:
        get_llm_provider(settings(), provider="x" * 5 + "<script>")
    assert "<script>" not in exc.value.message


def test_settings_reject_unknown_provider_value():
    with pytest.raises(ValidationError):
        settings(llm_provider="openai")


@pytest.mark.parametrize("key", [None, "", "   "])
def test_gemini_requires_key(key, monkeypatch):
    # A key sitting in the process environment must NOT be picked up implicitly.
    monkeypatch.setenv("GOOGLE_API_KEY", FAKE_KEY)
    kwargs = {} if key is None else {"gemini_api_key": key}
    with pytest.raises(LLMConfigurationError) as exc:
        get_llm_provider(settings(llm_provider="gemini", **kwargs))
    assert exc.value.message == "GEMINI_API_KEY is required when the provider is gemini."


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"ollama_base_url": "localhost:11434"}, "OLLAMA_BASE_URL"),
        ({"ollama_base_url": "ftp://x"}, "OLLAMA_BASE_URL"),
        ({"ollama_model": ""}, "valid model name"),
        ({"ollama_model": "bad model name!"}, "valid model name"),
    ],
)
def test_invalid_configuration(overrides, fragment):
    with pytest.raises(LLMConfigurationError, match=fragment):
        get_llm_provider(settings(**overrides))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_max_retries", 3),
        ("llm_max_retries", -1),
        ("llm_timeout_seconds", 0),
        ("llm_timeout_seconds", 0.5),
        ("llm_timeout_seconds", -1),
        ("llm_timeout_seconds", 301),
    ],
)
def test_timeout_and_retry_bounds(field, value):
    with pytest.raises(ValidationError):
        settings(**{field: value})


@pytest.mark.parametrize("value", [1, 60, 300])
def test_timeout_accepts_1_to_300_seconds(value):
    p = get_llm_provider(settings(llm_timeout_seconds=value))
    assert p._chat_model.client_kwargs["timeout"].read == value


def test_default_timeout_is_60_seconds():
    assert settings().llm_timeout_seconds == 60


def test_config_boundary_rejects_sub_second_timeout_even_if_settings_are_bypassed():
    s = settings().model_copy(update={"llm_timeout_seconds": 0.5})
    with pytest.raises(LLMConfigurationError, match="between 1 and 300"):
        LLMConfig.from_settings(s)


def test_config_never_exposes_the_key():
    cfg = LLMConfig.from_settings(settings(llm_provider="gemini", gemini_api_key=FAKE_KEY))
    p = get_llm_provider(settings(llm_provider="gemini", gemini_api_key=FAKE_KEY))
    for text in (repr(cfg), str(cfg), repr(p._chat_model), str(p._chat_model.model_dump())):
        assert FAKE_KEY not in text


def test_ollama_config_carries_no_key():
    cfg = LLMConfig.from_settings(settings(gemini_api_key=FAKE_KEY))
    assert cfg.provider == "ollama" and cfg.gemini_api_key is None


def test_provider_specifics_do_not_leak_outside_the_factory():
    """Only the LLM factory/classifier and (Step 8) the embedding provider module may know
    about concrete provider SDKs."""
    root = Path(__file__).resolve().parents[2]
    allowed = {
        "app/agent/llm/factory.py",
        "app/agent/llm/classify.py",
        "app/knowledge/embeddings/provider.py",
    }
    pattern = re.compile(
        r"^\s*(from|import)\s+(langchain_ollama|langchain_google_genai|ollama|google)\b", re.M
    )
    offenders = [
        str(p.relative_to(root))
        for base in ("app", "scripts")
        for p in (root / base).rglob("*.py")
        if pattern.search(p.read_text()) and str(p.relative_to(root)) not in allowed
    ]
    assert offenders == []
