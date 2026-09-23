"""Structured-output test vehicle: classify a message into an IntentAnalysis.

This proves the provider layer returns validated, machine-readable output. It is NOT the
production router and it does not call any tools.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.agent.llm.errors import LLMInputError
from app.agent.llm.provider import LLMProvider
from app.agent.prompts import intent as intent_prompt

MAX_INPUT_CHARS = 2000
MAX_ENTITIES = 10

IntentLabel = Literal[
    "customer_lookup",
    "order_lookup",
    "invoice_lookup",
    "shipment_lookup",
    "product_lookup",
    "general",
]
Entity = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class IntentAnalysis(BaseModel):
    """Classification of one user message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: IntentLabel = Field(description="The single best-matching intent.")
    entities: list[Entity] = Field(
        default_factory=list,
        max_length=MAX_ENTITIES,
        description="Identifiers or names mentioned in the message, copied exactly.",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0 to 1.")


class IntentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    analysis: IntentAnalysis
    provider: str
    model: str
    prompt_version: str
    attempts: int
    latency_ms: float


def analyze_intent(provider: LLMProvider, text: str) -> IntentResult:
    cleaned = text.strip() if isinstance(text, str) else ""
    if not cleaned:
        raise LLMInputError("The message to analyse is empty.")
    if len(cleaned) > MAX_INPUT_CHARS:
        raise LLMInputError(f"The message to analyse exceeds {MAX_INPUT_CHARS} characters.")
    result = provider.invoke_structured(
        IntentAnalysis,
        intent_prompt.build_messages(cleaned),
        operation=intent_prompt.PROMPT_ID,
        prompt_version=intent_prompt.PROMPT_VERSION,
    )
    return IntentResult(
        analysis=result.value,
        provider=provider.info.provider,
        model=provider.info.model,
        prompt_version=intent_prompt.PROMPT_VERSION,
        attempts=result.attempts,
        latency_ms=result.duration_ms,
    )
