"""Stable, safe assistant-level errors."""

from app.agent.assistant.result import InvalidToolCallSummary, RetrievalSummary, ToolCallSummary

MESSAGES = {
    "agent_input_invalid": "The request text is empty or too long.",
    "agent_limit_exceeded": "The assistant reached its tool-calling limit before finishing.",
    "agent_protocol_error": "The model produced an invalid tool request.",
    "agent_empty_answer": "The model returned an empty answer.",
    "agent_retrieval_error": "Policy knowledge could not be retrieved.",
    "agent_grounding_error": "The answer could not be grounded in retrieved policy sources.",
}


class AssistantError(Exception):
    """Raised when a run cannot produce an answer. ``code`` is stable; ``message`` is safe.

    LLM-layer failures keep their Step 4 code (e.g. ``llm_timeout``). Partial metadata is
    attached for later evaluation; it never contains raw tool payloads or reasoning.
    """

    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        detail: str | None = None,
        model_calls: int = 0,
        tool_calls: list[ToolCallSummary] | None = None,
        invalid_tool_calls: list[InvalidToolCallSummary] | None = None,
        retrievals: list[RetrievalSummary] | None = None,
    ) -> None:
        self.code = code
        self.message = message or MESSAGES.get(
            code, "The assistant could not complete the request."
        )
        self.detail = detail  # safe internal classification, e.g. "max_tool_calls"
        self.model_calls = model_calls
        self.tool_calls = list(tool_calls or [])
        self.invalid_tool_calls = list(invalid_tool_calls or [])
        self.retrievals = list(retrievals or [])  # LangGraph RAG path only
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
