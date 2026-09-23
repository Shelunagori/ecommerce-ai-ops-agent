"""Map provider/framework exceptions to typed, safe LLM errors.

Order matters: LangChain's normalised ``ModelError`` hierarchy first (the Gemini
integration raises these), then provider SDK errors, then transport errors.
"""

import json
import socket

import httpx
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from pydantic import ValidationError

from app.agent.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMInternalError,
    LLMOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
)

try:  # optional SDKs: only present when the integrations are installed
    from ollama import ResponseError as OllamaResponseError
except ImportError:  # pragma: no cover
    OllamaResponseError = None  # type: ignore[assignment,misc]
try:
    from google.genai.errors import APIError as GoogleAPIError
except ImportError:  # pragma: no cover
    GoogleAPIError = None  # type: ignore[assignment,misc]


def _model_not_found(provider: str | None, model: str | None) -> LLMConfigurationError:
    hint = f" Run: ollama pull {model}" if provider == "ollama" and model else ""
    return LLMConfigurationError(
        f"The configured model is not available from the provider.{hint}",
        code="llm_model_not_found",
    )


def _by_status(status: int | None, provider: str | None, model: str | None) -> LLMError:
    if status in (401, 403):
        return LLMAuthenticationError()
    if status == 404:
        return _model_not_found(provider, model)
    if status == 408:
        return LLMTimeoutError()
    if status == 429:
        return LLMUnavailableError(
            "The language model provider is rate limiting requests.", code="llm_rate_limited"
        )
    if status is not None and status >= 500:
        return LLMUnavailableError()
    return LLMInternalError()


def classify_exception(
    exc: BaseException, *, provider: str | None = None, model: str | None = None
) -> LLMError:
    err = _classify(exc, provider, model)
    err.provider = provider
    err.error_type = err.error_type or type(exc).__name__
    return err


def _classify(exc: BaseException, provider: str | None, model: str | None) -> LLMError:
    if isinstance(exc, LLMError):
        return exc
    # LangChain-normalised model errors (provider-neutral).
    if isinstance(exc, ModelError):
        if isinstance(exc, (ModelAuthenticationError, ModelPermissionDeniedError)):
            return LLMAuthenticationError()
        if isinstance(exc, ModelNotFoundError):
            return _model_not_found(provider, model)
        if isinstance(exc, ModelTimeoutError):
            return LLMTimeoutError()
        if isinstance(exc, ModelRateLimitError):
            return _by_status(429, provider, model)
        if isinstance(exc, (ModelConnectionError, ModelAPIError)):
            return LLMUnavailableError()
        if isinstance(exc, (ModelInvalidRequestError, ContextOverflowError)):
            return LLMInternalError(
                "The language model rejected the request.", code="llm_request_rejected"
            )
        return LLMUnavailableError() if exc.is_retryable else LLMInternalError()
    # Structured-output parsing / validation.
    if isinstance(exc, (OutputParserException, ValidationError, json.JSONDecodeError)):
        return LLMOutputError()
    # Provider SDK errors carrying an HTTP status.
    if OllamaResponseError is not None and isinstance(exc, OllamaResponseError):
        return _by_status(getattr(exc, "status_code", None), provider, model)
    if GoogleAPIError is not None and isinstance(exc, GoogleAPIError):
        return _by_status(getattr(exc, "code", None), provider, model)
    if isinstance(exc, httpx.HTTPStatusError):
        return _by_status(exc.response.status_code, provider, model)
    # Transport.
    if isinstance(exc, (httpx.TimeoutException, TimeoutError, socket.timeout)):
        return LLMTimeoutError()
    if isinstance(exc, (httpx.TransportError, ConnectionError, OSError)):
        return LLMUnavailableError()
    return LLMInternalError()
