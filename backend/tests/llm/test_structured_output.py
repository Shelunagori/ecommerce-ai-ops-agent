"""Structured-output contract and the provider-neutral analyze_intent service."""

import pytest
from langchain_core.exceptions import OutputParserException

from app.agent.llm import LLMInputError, LLMOutputError
from app.agent.llm.intent import MAX_INPUT_CHARS, IntentAnalysis, analyze_intent
from app.agent.prompts import intent as intent_prompt
from tests.llm.conftest import make_provider, ok

GOOD = {"intent": "order_lookup", "entities": ["ORD-1001"], "confidence": 0.98}


def test_valid_structured_result():
    provider, chat = make_provider(ok(GOOD))
    result = analyze_intent(provider, "Show me order ORD-1001")
    assert result.analysis == IntentAnalysis(**GOOD)
    assert (result.provider, result.model, result.attempts) == ("fake", "fake-model", 1)
    assert result.prompt_version == intent_prompt.PROMPT_VERSION == "intent-v1"
    call = chat.calls[0]
    assert call["method"] == "json_schema" and call["include_raw"] is True
    assert isinstance(call["schema"], dict) and call["schema"]["additionalProperties"] is False
    assert "maxLength" not in str(call["schema"])  # non-portable keywords stripped for providers
    assert call["messages"][0].content == intent_prompt.SYSTEM_PROMPT
    assert "<message>\nShow me order ORD-1001\n</message>" == call["messages"][1].content


def test_parsed_pydantic_instance_is_accepted():
    provider, _ = make_provider(ok(IntentAnalysis(**GOOD)))
    assert analyze_intent(provider, "ORD-1001").analysis.intent == "order_lookup"


def test_entities_default_to_empty_and_are_trimmed():
    provider, _ = make_provider(ok({"intent": "general", "confidence": 0.9}))
    assert analyze_intent(provider, "hello").analysis.entities == []
    provider, _ = make_provider(ok({**GOOD, "entities": ["  ORD-1001 "]}))
    assert analyze_intent(provider, "x").analysis.entities == ["ORD-1001"]


@pytest.mark.parametrize(
    "parsed",
    [
        {**GOOD, "tenant_id": "abc"},  # extra field
        {**GOOD, "reasoning": "because"},  # extra field
        {**GOOD, "confidence": 1.5},
        {**GOOD, "confidence": -0.01},
        {**GOOD, "confidence": "high"},
        {**GOOD, "entities": [f"E-{i}" for i in range(11)]},
        {**GOOD, "entities": ["X" * 65]},
        {**GOOD, "entities": ["   "]},
        {**GOOD, "intent": "refund_customer"},
        {"entities": [], "confidence": 0.5},  # missing intent
        [],
        "order_lookup",
    ],
)
def test_invalid_structured_output_is_rejected_without_retry(parsed):
    provider, chat = make_provider(ok(parsed), retries=2)
    with pytest.raises(LLMOutputError) as exc:
        analyze_intent(provider, "Show me order ORD-1001")
    assert exc.value.code == "llm_output_invalid"
    assert len(chat.calls) == 1  # J4: not retried


@pytest.mark.parametrize(
    "raw_result",
    [
        {"raw": object(), "parsed": None, "parsing_error": OutputParserException("bad json")},
        {"raw": object(), "parsed": None, "parsing_error": None},  # empty response
        "not-a-dict",
        None,
    ],
)
def test_malformed_or_empty_model_output(raw_result):
    provider, chat = make_provider(raw_result, retries=2)
    with pytest.raises(LLMOutputError):
        analyze_intent(provider, "Show me order ORD-1001")
    assert len(chat.calls) == 1


def test_parser_exception_raised_by_runnable_is_output_error():
    provider, _ = make_provider(OutputParserException("Invalid json output: {oops"))
    with pytest.raises(LLMOutputError):
        analyze_intent(provider, "ORD-1001")


@pytest.mark.parametrize("text", ["", "   ", "\n\t", "x" * (MAX_INPUT_CHARS + 1), None])
def test_invalid_input_is_rejected_before_calling_the_model(text):
    provider, chat = make_provider(ok(GOOD))
    with pytest.raises(LLMInputError) as exc:
        analyze_intent(provider, text)  # type: ignore[arg-type]
    assert exc.value.code == "llm_input_invalid"
    assert chat.calls == []


def test_input_at_limit_is_allowed():
    provider, _ = make_provider(ok(GOOD))
    analyze_intent(provider, "x" * MAX_INPUT_CHARS)


def test_works_with_any_protocol_implementation():
    """Callers depend on the LLMProvider protocol only - no provider branching."""
    from app.agent.llm.provider import ProviderInfo, StructuredResult

    class StaticProvider:
        info = ProviderInfo(provider="static", model="none")

        def invoke_structured(self, schema, messages, *, operation, prompt_version=None):
            return StructuredResult(schema(**GOOD), attempts=1, duration_ms=0.1)

    assert analyze_intent(StaticProvider(), "ORD-1001").provider == "static"


def test_prompt_module_is_versioned_and_separate():
    assert intent_prompt.PROMPT_ID == "intent_analysis"
    assert intent_prompt.PROMPT_VERSION.startswith("intent-v")
    for label in (
        "customer_lookup",
        "order_lookup",
        "invoice_lookup",
        "shipment_lookup",
        "product_lookup",
        "general",
    ):
        assert label in intent_prompt.SYSTEM_PROMPT
