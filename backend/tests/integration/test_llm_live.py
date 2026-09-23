"""OPT-IN live model tests. Skipped unless explicitly enabled:

    RUN_OLLAMA_INTEGRATION=1  (needs `ollama serve` + `ollama pull llama3.2:3b`)
    RUN_GEMINI_INTEGRATION=1  (needs GEMINI_API_KEY; synthetic prompts only; may cost quota)

The normal suite never makes external model calls.
"""

import os

import pytest

from app.agent.llm import get_llm_provider
from app.agent.llm.intent import analyze_intent

CASES = [
    ("Show order ORD-1001", "order_lookup", "ORD-1001"),
    ("What invoices does customer CUS-1001 have?", "invoice_lookup", "CUS-1001"),
    ("Where is shipment SHP-1003?", "shipment_lookup", "SHP-1003"),
    ("hello", "general", None),
]


def _enabled(flag: str) -> bool:
    return os.getenv(flag) == "1"


@pytest.mark.llm_integration
@pytest.mark.parametrize(
    "provider_name",
    [
        pytest.param(
            "ollama",
            marks=pytest.mark.skipif(
                not _enabled("RUN_OLLAMA_INTEGRATION"), reason="set RUN_OLLAMA_INTEGRATION=1"
            ),
        ),
        pytest.param(
            "gemini",
            marks=pytest.mark.skipif(
                not _enabled("RUN_GEMINI_INTEGRATION"), reason="set RUN_GEMINI_INTEGRATION=1"
            ),
        ),
    ],
)
@pytest.mark.parametrize(("text", "intent", "entity"), CASES)
def test_live_structured_intent(provider_name, text, intent, entity):
    provider = get_llm_provider(provider=provider_name)
    result = analyze_intent(provider, text)  # schema-validated or raises
    assert result.analysis.intent == intent
    if entity:
        assert entity in result.analysis.entities
    assert 0.0 <= result.analysis.confidence <= 1.0
