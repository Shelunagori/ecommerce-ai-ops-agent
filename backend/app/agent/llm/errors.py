"""Typed, safe errors for the LLM provider layer.

Every error carries a stable ``code`` and a ``message`` that is safe to show to API /
model consumers. Raw provider exception text is never stored in ``message``; only the
original exception *type name* is kept (``error_type``) for logs.
"""


class LLMError(Exception):
    code = "llm_internal_error"
    message = "The language model request failed unexpectedly."
    retryable = False

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        retryable: bool | None = None,
        provider: str | None = None,
        error_type: str | None = None,
    ) -> None:
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        self.provider = provider
        self.error_type = error_type
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class LLMConfigurationError(LLMError):
    """Missing/invalid configuration (unknown provider, missing key, model not pulled)."""

    code = "llm_not_configured"
    message = "The language model is not configured correctly."


class LLMAuthenticationError(LLMConfigurationError):
    code = "llm_auth_failed"
    message = "The language model provider rejected the credentials."


class LLMUnavailableError(LLMError):
    """Provider unreachable, overloaded, rate limited or returning server errors."""

    code = "llm_unavailable"
    message = "The language model provider is currently unavailable."
    retryable = True


class LLMTimeoutError(LLMError):
    code = "llm_timeout"
    message = "The language model did not respond in time."
    retryable = True


class LLMOutputError(LLMError):
    """The model answered, but not with valid structured output. Not retried (J4)."""

    code = "llm_output_invalid"
    message = "The language model returned an invalid structured response."


class LLMInputError(LLMError):
    """The caller supplied input that will not be sent (e.g. empty or too long)."""

    code = "llm_input_invalid"
    message = "The input for the language model is invalid."


class LLMInternalError(LLMError):
    code = "llm_internal_error"
