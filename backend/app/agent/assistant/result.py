"""Application-level result contract for one assistant run.

Contains only safe, evaluation-friendly metadata: no reasoning/thinking content, no raw
tool payloads, no tenant identifiers, no secrets or internal exception text.
"""

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
    reason: Literal["unparseable", "missing_id", "duplicate_id"]


class AssistantResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    provider: str
    model: str
    prompt_version: str
    model_calls: int
    tool_calls: list[ToolCallSummary]
    duration_ms: float
