"""Provider-neutral model interface and the single LangChain-backed implementation.

Callers depend on the ``LLMProvider`` protocol only. ``ChatModelProvider`` wraps any
LangChain chat model; everything provider-specific lives in ``factory.py`` and
``classify.py``.
"""

import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
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


class LLMProvider(Protocol):
    """What CommerceOps needs from a model today: validated structured output."""

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


class ChatModelProvider:
    def __init__(
        self,
        chat_model: BaseChatModel,
        info: ProviderInfo,
        retry: RetryPolicy,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._chat_model = chat_model
        self._info = info
        self._retry = retry
        self._sleep = sleep

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
        started = time.perf_counter()
        input_chars = sum(len(str(m.content)) for m in messages)
        attempt = 0
        while True:
            attempt += 1
            try:
                value = self._attempt(schema, messages)
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
            return StructuredResult(value=value, attempts=attempt, duration_ms=duration)

    def _attempt(self, schema: type[T], messages: Sequence[BaseMessage]) -> T:
        runnable = self._chat_model.with_structured_output(
            provider_json_schema(schema), method="json_schema", include_raw=True
        )
        out: Any = runnable.invoke(list(messages))
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
