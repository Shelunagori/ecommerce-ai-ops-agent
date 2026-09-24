"""Application-level result contract for one assistant run.

Contains only safe, evaluation-friendly metadata: no reasoning/thinking content, no raw
tool payloads, no tenant identifiers, no secrets or internal exception text.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict

ToolOutcome = Literal[
    "success",
    "not_found",
    "invalid_arguments",
    "unknown_tool",
    "service_unavailable",
    "internal_error",
]

ArgumentValue = str | int | float | bool | None


class ToolCallSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    round: int
    tool: str
    arguments: dict[str, ArgumentValue]  # only schema-declared business arguments
    rejected_argument_names: list[str] = []  # e.g. a smuggled "tenant_id" (name only)
    outcome: ToolOutcome
    error_code: str | None = None
    duration_ms: float


class InvalidToolCallSummary(BaseModel):
    """A model tool request the provider could not parse. Never executed."""

    model_config = ConfigDict(frozen=True)

    round: int
    name: str | None
    reason: Literal[
        "unparseable",  # AIMessage.invalid_tool_calls
        "missing_id",
        "duplicate_id",
        "textual_tool_call",  # pseudo tool call written as text (JSON / <tool_call> markup)
        "empty_structured_output",  # bare {} / [] instead of an answer
    ]


class RetrievalSummary(BaseModel):
    """One model-requested policy retrieval (LangGraph RAG path, Step 9).

    Deliberately WITHOUT the model's query text. ``as_of`` is the effective date actually
    used (None when the arguments were invalid and nothing ran); ``citations`` are every
    chunk returned to the model, in rank order.
    """

    model_config = ConfigDict(frozen=True)

    round: int
    as_of: date | None
    result_count: int
    citations: list[str]
    outcome: Literal["success", "no_results", "invalid_arguments", "error"]
    error_code: str | None = None
    rejected_argument_names: list[str] = []
    duration_ms: float


class PolicyCitation(BaseModel):
    """A policy source the FINAL answer actually cites (not every retrieved chunk)."""

    model_config = ConfigDict(frozen=True)

    citation: str  # policy://<document_key>/v<version>#chunk-<index>
    title: str
    document_key: str
    version: int
    section: str
    effective_from: date
    effective_to: date | None


class AssistantResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    provider: str
    model: str
    prompt_version: str
    model_calls: int
    tool_calls: list[ToolCallSummary]
    duration_ms: float
    # Step 9 (LangGraph RAG path). Default-empty: Step-5 callers are unaffected.
    retrievals: list[RetrievalSummary] = []
    citations: list[PolicyCitation] = []
