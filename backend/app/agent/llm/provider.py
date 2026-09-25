"""Provider-neutral model interface, the LangChain-backed implementation and the fallback.

Callers depend on the ``LLMProvider`` protocol only. ``ChatModelProvider`` wraps any
LangChain chat model; everything provider-specific lives in ``factory.py`` and
``classify.py``. ``FallbackProvider`` composes two providers at the boundary of ONE model
call (see its docstring).
"""

import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, Literal, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from app.agent.llm.classify import classify_exception
from app.agent.llm.errors import LLMError, LLMOutputError

logger = logging.getLogger("app.agent.llm")

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ProviderInfo:
    provider: str
    model: str


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    value: T
    attempts: int
    duration_ms: float
    provider: str | None = None  # the provider that ACTUALLY produced the value
    model: str | None = None
    fallback_used: bool = False


@dataclass(frozen=True)
class ChatResult:
    message: AIMessage
    attempts: int
    duration_ms: float
    provider: str | None = None  # the provider that ACTUALLY answered this model call
    model: str | None = None
    fallback_used: bool = False  # True when the primary failed and the fallback answered


class LLMProvider(Protocol):
    """What CommerceOps needs from a model: validated structured output and tool-calling
    chat turns. Provider specifics stay behind this interface."""

    @property
    def info(self) -> ProviderInfo: ...

    def invoke_structured(
        self,
        schema: type[T],
        messages: Sequence[BaseMessage],
        *,
        operation: str,
        prompt_version: str | None = None,
    ) -> StructuredResult[T]: ...

    def invoke_chat(
        self,
        messages: Sequence[BaseMessage],
        *,
        tools: Sequence[BaseTool],
        operation: str,
        prompt_version: str | None = None,
    ) -> ChatResult: ...


@dataclass(frozen=True)
class RetryPolicy:
    """Retry only transient failures (timeout / unavailable), a small number of times."""

    max_retries: int = 1
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0

    def delay(self, retry_number: int) -> float:
        capped = min(self.max_delay_seconds, self.base_delay_seconds * 2 ** (retry_number - 1))
        return capped * random.uniform(0.8, 1.2)  # noqa: S311 - jitter, not crypto


# Keywords outside the portable subset both providers document for native structured output
# (Gemini lists no string-length/pattern support). They stay enforced locally because every
# response is re-validated with the full Pydantic schema.
_PROVIDER_UNSUPPORTED_KEYWORDS = frozenset({"minLength", "maxLength", "pattern"})


def provider_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """JSON schema sent to the provider: the Pydantic schema minus non-portable keywords."""

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in _PROVIDER_UNSUPPORTED_KEYWORDS}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return strip(schema.model_json_schema())


StructuredMethod = Literal["json_schema", "tool_call"]
MessagePreparer = Callable[[Sequence[BaseMessage]], list[BaseMessage]]


class ChatModelProvider:
    def __init__(
        self,
        chat_model: BaseChatModel,
        info: ProviderInfo,
        retry: RetryPolicy,
        *,
        sleep: Callable[[float], None] = time.sleep,
        structured_method: StructuredMethod = "json_schema",
        prepare_messages: MessagePreparer | None = None,
    ) -> None:
        """``structured_method``: provider-native JSON schema output (default), or one forced
        tool call whose arguments are the object (providers without JSON-schema mode).
        ``prepare_messages``: provider-specific history normalisation applied right before
        sending (e.g. drop another provider's metadata); graph state is never changed."""
        self._chat_model = chat_model
        self._info = info
        self._retry = retry
        self._sleep = sleep
        self._structured_method = structured_method
        self._prepare = prepare_messages or list

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def invoke_structured(
        self,
        schema: type[T],
        messages: Sequence[BaseMessage],
        *,
        operation: str,
        prompt_version: str | None = None,
    ) -> StructuredResult[T]:
        value, attempts, duration = self._with_retry(
            lambda: self._attempt(schema, messages), messages, operation, prompt_version
        )
        return StructuredResult(
            value=value,
            attempts=attempts,
            duration_ms=duration,
            provider=self._info.provider,
            model=self._info.model,
        )

    def invoke_chat(
        self,
        messages: Sequence[BaseMessage],
        *,
        tools: Sequence[BaseTool],
        operation: str,
        prompt_version: str | None = None,
    ) -> ChatResult:
        """One chat turn with ``tools`` bound. Returns the provider's AIMessage unchanged
        (tool calls and provider metadata such as Gemini thought signatures included)."""
        value, attempts, duration = self._with_retry(
            lambda: self._chat_attempt(messages, tools), messages, operation, prompt_version
        )
        return ChatResult(
            message=value,
            attempts=attempts,
            duration_ms=duration,
            provider=self._info.provider,
            model=self._info.model,
        )

    def _with_retry(
        self,
        attempt_fn: Callable[[], Any],
        messages: Sequence[BaseMessage],
        operation: str,
        prompt_version: str | None,
    ) -> tuple[Any, int, float]:
        started = time.perf_counter()
        input_chars = sum(len(str(m.content)) for m in messages)
        attempt = 0
        while True:
            attempt += 1
            try:
                value = attempt_fn()
            except Exception as exc:  # noqa: BLE001 - classified below, never re-raised raw
                err = classify_exception(exc, provider=self._info.provider, model=self._info.model)
                if err.retryable and attempt <= self._retry.max_retries:
                    self._log(
                        operation, prompt_version, "retrying", err, attempt, started, input_chars
                    )
                    self._sleep(self._retry.delay(attempt))
                    continue
                self._log(operation, prompt_version, "error", err, attempt, started, input_chars)
                raise err from None
            duration = round((time.perf_counter() - started) * 1000, 1)
            self._log(operation, prompt_version, "ok", None, attempt, started, input_chars)
            return value, attempt, duration

    def _chat_attempt(
        self, messages: Sequence[BaseMessage], tools: Sequence[BaseTool]
    ) -> AIMessage:
        runnable = self._chat_model.bind_tools(list(tools)) if tools else self._chat_model
        out = runnable.invoke(self._prepare(messages))
        if not isinstance(out, AIMessage):
            raise LLMOutputError(error_type="unexpected_message_type")
        return out

    def _attempt(self, schema: type[T], messages: Sequence[BaseMessage]) -> T:
        if self._structured_method == "tool_call":
            return self._tool_call_attempt(schema, messages)
        runnable = self._chat_model.with_structured_output(
            provider_json_schema(schema), method="json_schema", include_raw=True
        )
        out: Any = runnable.invoke(self._prepare(messages))
        if not isinstance(out, dict):
            raise LLMOutputError(error_type="unexpected_result_shape")
        if out.get("parsing_error") is not None or out.get("parsed") is None:
            raise LLMOutputError(
                error_type="parsing_error" if out.get("parsing_error") else "empty"
            )
        parsed = out["parsed"]
        data = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
        try:
            # Re-validate with OUR schema (extra="forbid", bounds) regardless of provider parser.
            return schema.model_validate(data)
        except ValidationError:
            raise LLMOutputError(error_type="schema_validation") from None

    def _tool_call_attempt(self, schema: type[T], messages: Sequence[BaseMessage]) -> T:
        """Structured output as exactly one forced call of a single schema-shaped tool."""
        name = schema.__name__
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": (schema.__doc__ or name).strip()[:500],
                "parameters": provider_json_schema(schema),
            },
        }
        out = self._chat_model.bind_tools([tool], tool_choice="required").invoke(
            self._prepare(messages)
        )
        calls = getattr(out, "tool_calls", None) or []
        if not isinstance(out, AIMessage) or len(calls) != 1 or calls[0].get("name") != name:
            raise LLMOutputError(error_type="no_structured_tool_call")
        try:
            # Same strict re-validation as the JSON-schema path (extra="forbid", bounds).
            return schema.model_validate(calls[0].get("args"))
        except ValidationError:
            raise LLMOutputError(error_type="schema_validation") from None

    def _log(
        self,
        operation: str,
        prompt_version: str | None,
        outcome: str,
        err: LLMError | None,
        attempts: int,
        started: float,
        input_chars: int,
    ) -> None:
        # Never logs prompts, responses or credentials: identifiers and classifications only.
        fields: dict[str, Any] = {
            "provider": self._info.provider,
            "model": self._info.model,
            "operation": operation,
            "outcome": outcome,
            "attempts": attempts,
            "input_chars": input_chars,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if prompt_version:
            fields["prompt_version"] = prompt_version
        if err is not None:
            fields["error_code"] = err.code
            fields["error_type"] = err.error_type
        level = logging.INFO if outcome == "ok" else logging.WARNING
        logger.log(level, "llm call", extra=fields)


# Availability failures of the PRIMARY provider that another provider may absorb. Everything
# else - invalid/unparseable output (llm_output_invalid), rejected requests, credentials or
# configuration, and every application/protocol/grounding/tenant/action error raised AFTER a
# model call - is never retried on another provider.
FALLBACK_ERROR_CODES = frozenset(
    {"llm_rate_limited", "llm_quota_exceeded", "llm_unavailable", "llm_timeout"}
)


class FallbackProvider:
    """Primary provider with an optional fallback for ONE model call.

    The boundary is a single ``invoke_chat`` / ``invoke_structured`` call: the primary (after
    its own RetryPolicy) fails with an availability error -> the fallback receives the SAME
    messages and tools and answers THAT call. Nothing earlier is re-run: tools, retrieval,
    proposals and executions already happened in graph nodes outside this call, and the
    graph continues from the fallback's answer. The result names the provider that actually
    answered (``ChatResult.provider`` / ``fallback_used``) so traces never misattribute it.
    """

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        if primary.info.provider == fallback.info.provider:
            raise ValueError("the fallback provider must differ from the primary provider")
        self._primary = primary
        self._fallback = fallback

    @property
    def info(self) -> ProviderInfo:
        return self._primary.info  # the configured primary (per-call truth is on the result)

    @property
    def fallback_info(self) -> ProviderInfo:
        return self._fallback.info

    def invoke_chat(
        self,
        messages: Sequence[BaseMessage],
        *,
        tools: Sequence[BaseTool],
        operation: str,
        prompt_version: str | None = None,
    ) -> ChatResult:
        kw = {"tools": tools, "operation": operation, "prompt_version": prompt_version}
        try:
            return _stamped(self._primary.invoke_chat(messages, **kw), self._primary.info)
        except LLMError as exc:
            if exc.code not in FALLBACK_ERROR_CODES:
                raise
            self._log_fallback(operation, exc)
        return replace(
            _stamped(self._fallback.invoke_chat(messages, **kw), self._fallback.info),
            fallback_used=True,
        )

    def invoke_structured(
        self,
        schema: type[T],
        messages: Sequence[BaseMessage],
        *,
        operation: str,
        prompt_version: str | None = None,
    ) -> StructuredResult[T]:
        kw = {"operation": operation, "prompt_version": prompt_version}
        try:
            return self._primary.invoke_structured(schema, messages, **kw)
        except LLMError as exc:
            if exc.code not in FALLBACK_ERROR_CODES:
                raise
            self._log_fallback(operation, exc)
        result = self._fallback.invoke_structured(schema, messages, **kw)
        return replace(
            result,
            provider=result.provider or self._fallback.info.provider,
            model=result.model or self._fallback.info.model,
            fallback_used=True,
        )

    def _log_fallback(self, operation: str, exc: LLMError) -> None:
        # Identifiers and the safe error code only: never messages, keys or provider bodies.
        logger.warning(
            "llm fallback",
            extra={
                "provider": self._primary.info.provider,
                "model": self._primary.info.model,
                "fallback_provider": self._fallback.info.provider,
                "fallback_model": self._fallback.info.model,
                "operation": operation,
                "error_code": exc.code,
            },
        )


def _stamped(result: ChatResult, info: ProviderInfo) -> ChatResult:
    if result.provider:
        return result
    return replace(result, provider=info.provider, model=info.model)
