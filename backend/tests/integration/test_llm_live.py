"""OPT-IN live provider smoke tests. Skipped unless explicitly enabled:

    RUN_OLLAMA_INTEGRATION=1  (needs `ollama serve` + `ollama pull qwen3:4b-instruct`)
    RUN_GEMINI_INTEGRATION=1  (needs GEMINI_API_KEY; synthetic prompts only; uses quota)

These validate PROVIDER INTEGRATION (invocation, native structured output, schema
validation, metadata, safe errors) - not model quality. Whether a model picks the
"right" intent for a prompt is measured later by the evaluation framework, so no test
here asserts a specific intent label or entity. The normal suite never calls a model.
"""

import json
import logging
import os
import typing

import pytest

from app.agent.llm import LLMError, get_llm_provider
from app.agent.llm.intent import MAX_ENTITIES, IntentAnalysis, IntentLabel, analyze_intent
from app.agent.prompts import intent as intent_prompt
from app.core.config import get_settings

# Smoke inputs only (synthetic). Expected labels intentionally NOT asserted.
SMOKE_INPUTS = [
    "Show order ORD-1001",
    "What invoices does customer CUS-1001 have?",
    "Where is shipment SHP-1003?",
    "hello",
]
ALLOWED_INTENTS = set(typing.get_args(IntentLabel))


def _flag(name: str) -> bool:
    return os.getenv(name) == "1"


PROVIDERS = [
    pytest.param(
        "ollama",
        marks=pytest.mark.skipif(
            not _flag("RUN_OLLAMA_INTEGRATION"), reason="set RUN_OLLAMA_INTEGRATION=1"
        ),
    ),
    pytest.param(
        "gemini",
        marks=pytest.mark.skipif(
            not _flag("RUN_GEMINI_INTEGRATION"), reason="set RUN_GEMINI_INTEGRATION=1"
        ),
    ),
]


def _secret_values() -> list[str]:
    key = get_settings().gemini_api_key
    return [key.get_secret_value()] if key and key.get_secret_value() else []


@pytest.mark.llm_integration
@pytest.mark.parametrize("provider_name", PROVIDERS)
@pytest.mark.parametrize("text", SMOKE_INPUTS)
def test_live_structured_output_contract(provider_name, text, caplog):
    provider = get_llm_provider(provider=provider_name)
    with caplog.at_level(logging.INFO, logger="app.agent.llm"):
        result = analyze_intent(provider, text)  # raises a typed LLMError on any failure

    # Provider metadata
    assert result.provider == provider_name == provider.info.provider
    assert result.model == provider.info.model and result.model
    assert result.prompt_version == intent_prompt.PROMPT_VERSION
    assert 1 <= result.attempts and result.latency_ms > 0

    # Structured output validates against the contract
    analysis = result.analysis
    assert isinstance(analysis, IntentAnalysis)
    IntentAnalysis.model_validate(analysis.model_dump())  # round-trips under extra="forbid"
    assert analysis.intent in ALLOWED_INTENTS
    assert len(analysis.entities) <= MAX_ENTITIES
    assert all(1 <= len(e) <= 64 for e in analysis.entities)
    assert 0.0 <= analysis.confidence <= 1.0

    # Nothing sensitive in the serialised result or logs
    dumped = json.dumps(result.model_dump(mode="json"))
    for secret in _secret_values():
        assert secret not in dumped and secret not in caplog.text
    assert text not in caplog.text  # prompts are not logged
    assert "llm call" in caplog.text


@pytest.mark.llm_integration
@pytest.mark.skipif(not _flag("RUN_OLLAMA_INTEGRATION"), reason="set RUN_OLLAMA_INTEGRATION=1")
def test_live_ollama_missing_model_is_a_safe_configuration_error():
    provider = get_llm_provider(provider="ollama", model="commerceops-no-such-model:0b")
    with pytest.raises(LLMError) as exc:
        analyze_intent(provider, "hello")
    assert exc.value.code == "llm_model_not_found"
    assert exc.value.__cause__ is None
    assert "ollama pull commerceops-no-such-model:0b" in exc.value.message


@pytest.mark.llm_integration
@pytest.mark.skipif(not _flag("RUN_GEMINI_INTEGRATION"), reason="set RUN_GEMINI_INTEGRATION=1")
def test_live_gemini_rejected_key_is_a_safe_auth_error(caplog):
    from app.core.config import Settings

    bogus = "AIzaInvalid-commerceops-test-key-000000000"
    settings = Settings(llm_provider="gemini", gemini_api_key=bogus)
    provider = get_llm_provider(settings)
    with caplog.at_level(logging.DEBUG), pytest.raises(LLMError) as exc:
        analyze_intent(provider, "hello")
    # Gemini reports a bad key as 401/403 or as 400 INVALID_ARGUMENT, depending on the API.
    assert exc.value.code in {"llm_auth_failed", "llm_request_rejected"}
    assert exc.value.__cause__ is None
    for text in (exc.value.message, str(exc.value), repr(exc.value), caplog.text):
        assert bogus not in text
