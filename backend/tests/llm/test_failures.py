"""Failure classification, retry policy and security of errors/logs (offline)."""

import json
import logging

import httpx
import pytest
from langchain_core.exceptions import ModelTimeoutError
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAIError,
    GoogleAuthenticationError,
    GoogleModelNotFoundError,
    GooglePermissionDeniedError,
    GoogleRateLimitError,
)
from ollama import ResponseError

from app.agent.llm import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMInternalError,
    LLMOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
    get_llm_provider,
)
from app.agent.llm.intent import analyze_intent
from scripts import run_llm
from tests.llm.conftest import FAKE_KEY, make_provider, ok, settings

GOOD = {"intent": "order_lookup", "entities": ["ORD-1001"], "confidence": 0.9}
REQ = httpx.Request("POST", "http://localhost:11434/api/chat")


@pytest.mark.parametrize(
    ("exc", "err_type", "code", "retried"),
    [
        (httpx.ReadTimeout("timed out", request=REQ), LLMTimeoutError, "llm_timeout", True),
        (TimeoutError(), LLMTimeoutError, "llm_timeout", True),
        (ModelTimeoutError("t"), LLMTimeoutError, "llm_timeout", True),
        (httpx.ConnectError("refused", request=REQ), LLMUnavailableError, "llm_unavailable", True),
        (
            ConnectionError("Failed to connect to Ollama"),
            LLMUnavailableError,
            "llm_unavailable",
            True,
        ),
        (ResponseError("server exploded", 500), LLMUnavailableError, "llm_unavailable", True),
        (ResponseError("busy", 503), LLMUnavailableError, "llm_unavailable", True),
        (GoogleRateLimitError("quota"), LLMUnavailableError, "llm_rate_limited", True),
        (
            ResponseError("model 'llama3.2:3b' not found", 404),
            LLMConfigurationError,
            "llm_model_not_found",
            False,
        ),
        (GoogleModelNotFoundError("404"), LLMConfigurationError, "llm_model_not_found", False),
        (
            GoogleAuthenticationError("401 API key not valid"),
            LLMAuthenticationError,
            "llm_auth_failed",
            False,
        ),
        (GooglePermissionDeniedError("403"), LLMAuthenticationError, "llm_auth_failed", False),
        (ResponseError("unauthorized", 401), LLMAuthenticationError, "llm_auth_failed", False),
        (ChatGoogleGenerativeAIError("weird"), LLMInternalError, "llm_internal_error", False),
        (ValueError("boom"), LLMInternalError, "llm_internal_error", False),
        (KeyError("x"), LLMInternalError, "llm_internal_error", False),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_failure_mapping_and_retry_decision(exc, err_type, code, retried):
    sleeps: list[float] = []
    provider, chat = make_provider(exc, retries=1, sleeps=sleeps)
    with pytest.raises(err_type) as caught:
        analyze_intent(provider, "Show me order ORD-1001")
    assert caught.value.code == code
    assert len(chat.calls) == (2 if retried else 1)
    assert len(sleeps) == (1 if retried else 0)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__ is True
    assert caught.value.error_type == type(exc).__name__


def test_model_not_found_message_has_actionable_ollama_hint():
    from app.agent.llm.classify import classify_exception

    err = classify_exception(
        ResponseError("not found", 404), provider="ollama", model="llama3.2:3b"
    )
    assert err.message.endswith("Run: ollama pull llama3.2:3b")


def test_transient_failure_then_success():
    provider, chat = make_provider(httpx.ConnectError("x", request=REQ), ok(GOOD), retries=1)
    result = analyze_intent(provider, "ORD-1001")
    assert result.attempts == 2 and len(chat.calls) == 2


def test_retry_count_and_backoff_are_bounded():
    sleeps: list[float] = []
    provider, chat = make_provider(TimeoutError(), retries=2, sleeps=sleeps)
    with pytest.raises(LLMTimeoutError):
        analyze_intent(provider, "ORD-1001")
    assert len(chat.calls) == 3
    assert len(sleeps) == 2 and 0.4 <= sleeps[0] <= 0.6 and 0.8 <= sleeps[1] <= 1.2  # exp + jitter


def test_zero_retries_means_single_attempt():
    provider, chat = make_provider(TimeoutError(), retries=0)
    with pytest.raises(LLMTimeoutError):
        analyze_intent(provider, "ORD-1001")
    assert len(chat.calls) == 1


# --- security ------------------------------------------------------------------------------------
SECRET_PROMPT = "customer ZETA-SECRET-PROMPT-777"
SECRET_RESPONSE = "ZETA-SECRET-RESPONSE-888"


def test_secrets_in_provider_errors_never_surface(caplog):
    leaky = httpx.ConnectError(
        f"https://generativelanguage.googleapis.com/?key={FAKE_KEY}", request=REQ
    )
    provider, _ = make_provider(leaky, retries=1)
    with caplog.at_level(logging.DEBUG), pytest.raises(LLMUnavailableError) as caught:
        analyze_intent(provider, SECRET_PROMPT)
    for text in (str(caught.value), caught.value.message, repr(caught.value), caplog.text):
        assert FAKE_KEY not in text
        assert "generativelanguage" not in text
    assert SECRET_PROMPT not in caplog.text


def test_prompts_and_responses_are_not_logged(caplog):
    provider, _ = make_provider(
        ok({"intent": "general", "entities": [SECRET_RESPONSE], "confidence": 0.5})
    )
    with caplog.at_level(logging.DEBUG):
        analyze_intent(provider, SECRET_PROMPT)
    assert SECRET_PROMPT not in caplog.text and SECRET_RESPONSE not in caplog.text
    [record] = [r for r in caplog.records if r.name == "app.agent.llm"]
    assert (record.outcome, record.operation, record.prompt_version) == (
        "ok",
        "intent_analysis",
        "intent-v1",
    )
    assert record.input_chars > 0 and record.attempts == 1


def test_error_log_line_is_classification_only(caplog):
    provider, _ = make_provider(OutputErrorFactory(), retries=0)
    with caplog.at_level(logging.INFO), pytest.raises(LLMOutputError):
        analyze_intent(provider, SECRET_PROMPT)
    [record] = [r for r in caplog.records if r.name == "app.agent.llm"]
    assert (record.outcome, record.error_code) == ("error", "llm_output_invalid")
    assert SECRET_RESPONSE not in caplog.text


def OutputErrorFactory():  # noqa: N802 - reads like a value in the parametrised style above
    return {"raw": SECRET_RESPONSE, "parsed": None, "parsing_error": ValueError(SECRET_RESPONSE)}


def test_cli_missing_key_prints_safe_error(monkeypatch, capsys):
    monkeypatch.setattr(
        run_llm,
        "get_llm_provider",
        lambda **kw: get_llm_provider(settings(llm_provider="gemini"), **kw),
    )
    assert run_llm.main(["--provider", "gemini", "--text", "hi"]) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert json.loads(out.err.strip().splitlines()[-1])["error"]["code"] == "llm_not_configured"


def test_cli_success_prints_result_json_only(monkeypatch, capsys):
    provider, _ = make_provider(ok(GOOD))
    monkeypatch.setattr(run_llm, "get_llm_provider", lambda **kw: provider)
    assert run_llm.main(["--provider", "ollama", "--text", "Show me order ORD-1001"]) == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["analysis"] == GOOD and result["prompt_version"] == "intent-v1"
    assert FAKE_KEY not in out.out + out.err


def test_cli_never_prints_a_configured_key(monkeypatch, capsys):
    monkeypatch.setattr(
        run_llm,
        "get_llm_provider",
        lambda **kw: get_llm_provider(settings(gemini_api_key=FAKE_KEY), **kw),
    )
    run_llm.main(["--provider", "gemini", "--model", "bad model!", "--text", "hi"])
    out = capsys.readouterr()
    assert FAKE_KEY not in out.out + out.err
