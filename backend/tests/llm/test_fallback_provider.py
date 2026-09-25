"""FallbackProvider: one model call, availability errors only, same messages, honest result."""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.llm import LLMConfigurationError, LLMError, get_llm_provider
from app.agent.llm.errors import (
    LLMAuthenticationError,
    LLMInternalError,
    LLMOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.agent.llm.intent import IntentAnalysis
from app.agent.llm.provider import (
    FALLBACK_ERROR_CODES,
    ChatModelProvider,
    ChatResult,
    FallbackProvider,
    ProviderInfo,
    RetryPolicy,
    StructuredResult,
)
from tests.assistant.fakes import ScriptedChatModel
from tests.llm.conftest import FAKE_KEY, settings

MESSAGES = [HumanMessage("Where is SHP-1003?")]


def scripted(name: str, *steps, retries: int = 1) -> tuple[ChatModelProvider, ScriptedChatModel]:
    model = ScriptedChatModel(list(steps))
    p = ChatModelProvider(
        model,  # type: ignore[arg-type]
        ProviderInfo(provider=name, model=f"{name}-model"),
        RetryPolicy(max_retries=retries),
        sleep=lambda _s: None,
    )
    return p, model


def rate_limited() -> LLMError:
    return LLMUnavailableError("rate limited", code="llm_rate_limited")


def call(p):
    return p.invoke_chat(MESSAGES, tools=[], operation="commerce_assistant", prompt_version="v3")


# --- A ----------------------------------------------------------------------------------------
def test_primary_success_never_touches_the_fallback():
    cf, cf_model = scripted("cloudflare", AIMessage(content="From Cloudflare."))
    gm, gm_model = scripted("gemini")
    result = call(FallbackProvider(cf, gm))
    assert (result.provider, result.fallback_used) == ("cloudflare", False)
    assert result.message.content == "From Cloudflare."
    assert gm_model.invocations == []


# --- B/C/D: availability failures fall back for THIS call ----------------------------------------
@pytest.mark.parametrize(
    ("errors", "primary_attempts"),
    [
        ([rate_limited(), rate_limited()], 2),  # 429 after the primary's own retry
        ([LLMUnavailableError(), LLMUnavailableError()], 2),  # temporary 5xx
        ([LLMTimeoutError(), LLMTimeoutError()], 2),  # timeout
        ([LLMUnavailableError("q", code="llm_quota_exceeded", retryable=False)], 1),  # quota
    ],
)
def test_availability_failures_fall_back_after_the_primary_retry_policy(errors, primary_attempts):
    cf, cf_model = scripted("cloudflare", *errors)
    gm, gm_model = scripted("gemini", AIMessage(content="From Gemini."))
    result = call(FallbackProvider(cf, gm))
    assert (result.provider, result.model, result.fallback_used) == ("gemini", "gemini-model", True)
    assert len(cf_model.invocations) == primary_attempts
    assert len(gm_model.invocations) == 1
    # the fallback receives the EXACT same history (nothing replayed, nothing dropped)
    assert gm_model.invocations[0].messages == cf_model.invocations[0].messages == MESSAGES


def test_transient_then_success_on_the_primary_does_not_fall_back():
    cf, _ = scripted("cloudflare", LLMUnavailableError(), AIMessage(content="ok"))
    gm, gm_model = scripted("gemini")
    result = call(FallbackProvider(cf, gm))
    assert (result.provider, result.attempts, result.fallback_used) == ("cloudflare", 2, False)
    assert gm_model.invocations == []


# --- E/G + everything that is not availability: never falls back -------------------------------
@pytest.mark.parametrize(
    "error",
    [
        LLMOutputError(),  # invalid / unparseable model output
        LLMAuthenticationError(),  # bad credentials: must surface, not be hidden
        LLMConfigurationError("x", code="llm_model_not_found"),
        LLMInternalError("rejected", code="llm_request_rejected"),
        LLMInternalError(),
    ],
)
def test_non_availability_errors_never_fall_back(error):
    cf, _ = scripted("cloudflare", error)
    gm, gm_model = scripted("gemini", AIMessage(content="should not be used"))
    with pytest.raises(LLMError) as info:
        call(FallbackProvider(cf, gm))
    assert info.value.code == error.code and gm_model.invocations == []


def test_the_fallback_set_is_exactly_the_availability_codes():
    assert FALLBACK_ERROR_CODES == {
        "llm_rate_limited",
        "llm_quota_exceeded",
        "llm_unavailable",
        "llm_timeout",
    }


def test_a_schema_invalid_tool_call_is_a_successful_provider_call_so_no_fallback():
    """Tool-call validation happens in the graph AFTER the model call returned: the provider
    boundary sees a normal answer and must not second-guess it on another provider."""
    bad = AIMessage(content="", tool_calls=[{"name": "no_such_tool", "args": {}, "id": "t1"}])
    cf, _ = scripted("cloudflare", bad)
    gm, gm_model = scripted("gemini")
    result = call(FallbackProvider(cf, gm))
    assert result.provider == "cloudflare" and gm_model.invocations == []


# --- J: both unavailable -----------------------------------------------------------------------
def test_both_unavailable_returns_the_safe_error():
    cf, _ = scripted("cloudflare", rate_limited(), rate_limited())
    gm, gm_model = scripted("gemini", LLMUnavailableError(), LLMUnavailableError())
    with pytest.raises(LLMError) as info:
        call(FallbackProvider(cf, gm))
    assert info.value.code == "llm_unavailable" and len(gm_model.invocations) == 2
    assert info.value.message == "The language model provider is currently unavailable."


def test_fallback_is_logged_with_safe_fields_only(caplog):
    caplog.set_level(logging.INFO, logger="app.agent.llm")
    cf, _ = scripted("cloudflare", rate_limited(), rate_limited())
    gm, _ = scripted("gemini", AIMessage(content="ok"))
    call(FallbackProvider(cf, gm))
    [record] = [r for r in caplog.records if r.getMessage() == "llm fallback"]
    fields = {k: getattr(record, k) for k in ("provider", "fallback_provider", "error_code")}
    assert fields == {
        "provider": "cloudflare",
        "fallback_provider": "gemini",
        "error_code": "llm_rate_limited",
    }
    assert "SHP-1003" not in caplog.text  # never message content


def test_structured_output_falls_back_the_same_way():
    class Structured:
        def __init__(self, name, outcome):
            self.info = ProviderInfo(name, f"{name}-m")
            self.outcome = outcome
            self.calls = 0

        def invoke_structured(self, schema, messages, **_):
            self.calls += 1
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return StructuredResult(value=self.outcome, attempts=1, duration_ms=1.0)

    value = IntentAnalysis(intent="shipment_lookup", entities=["SHP-1003"], confidence=0.9)
    fb = FallbackProvider(Structured("cloudflare", LLMTimeoutError()), Structured("gemini", value))
    result = fb.invoke_structured(IntentAnalysis, MESSAGES, operation="intent")
    assert (result.value, result.provider, result.fallback_used) == (value, "gemini", True)
    fb = FallbackProvider(Structured("cloudflare", LLMOutputError()), Structured("gemini", value))
    with pytest.raises(LLMOutputError):
        fb.invoke_structured(IntentAnalysis, MESSAGES, operation="intent")


def test_same_provider_twice_is_refused():
    a, _ = scripted("gemini")
    b, _ = scripted("gemini")
    with pytest.raises(ValueError):
        FallbackProvider(a, b)


# --- configuration --------------------------------------------------------------------------
CF = {
    "llm_provider": "cloudflare",
    "cloudflare_account_id": "0123456789abcdef0123456789abcdef",
    "cloudflare_api_token": "cf-token",
}


def test_factory_builds_cloudflare_primary_with_gemini_fallback(no_network):
    p = get_llm_provider(settings(**CF, llm_fallback_provider="gemini", gemini_api_key=FAKE_KEY))
    assert isinstance(p, FallbackProvider)
    assert (p.info.provider, p.fallback_info.provider) == ("cloudflare", "gemini")
    assert no_network == []


def test_no_fallback_configured_means_a_plain_provider():
    p = get_llm_provider(settings(**CF))
    assert isinstance(p, ChatModelProvider) and p.info.provider == "cloudflare"
    gemini_only = get_llm_provider(settings(llm_provider="gemini", gemini_api_key=FAKE_KEY))
    assert isinstance(gemini_only, ChatModelProvider)


def test_fallback_requires_its_own_credentials_and_must_differ():
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        get_llm_provider(settings(**CF, llm_fallback_provider="gemini"))
    with pytest.raises(LLMConfigurationError, match="must differ"):
        get_llm_provider(settings(**CF, llm_fallback_provider="cloudflare"))


def test_explicit_cli_override_never_adds_a_fallback():
    p = get_llm_provider(
        settings(**CF, llm_fallback_provider="gemini", gemini_api_key=FAKE_KEY),
        provider="cloudflare",
    )
    assert isinstance(p, ChatModelProvider)


def test_chat_result_names_the_provider_that_answered():
    r = ChatResult(message=AIMessage(content="x"), attempts=1, duration_ms=1.0)
    assert r.provider is None and r.fallback_used is False
