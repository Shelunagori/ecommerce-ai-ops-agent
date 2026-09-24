"""Hosted embeddings: Gemini API (Phase 8), behind the same ``EmbeddingProvider`` contract.

Verified against Google's current documentation (Sept 2026):

* model ``gemini-embedding-2`` (``gemini-embedding-001`` is legacy); 128-3072 output
  dimensions, 768 recommended; truncated dimensions are auto-normalised;
* retrieval inputs use documented TEXT PREFIXES instead of ``task_type``:
  documents ``"title: {title} | text: {content}"``, queries
  ``"task: search result | query: {content}"`` -> ``policy-embedding-input-gemini-v1``;
* several texts in ONE ``embed_content`` call are aggregated into one embedding unless each
  is wrapped in its own ``Content`` - so every input is wrapped;
* max 8 192 input tokens. The Developer API has no ``auto_truncate`` switch, so this provider
  REFUSES inputs longer than ``MAX_INPUT_CHARS`` (far below the token window) before any
  request: nothing can be truncated silently.

Construction is network-free (the SDK client is created lazily). The profile's model digest is
``sha256("gemini:<resolved model name>:<version>:<dimensions>")`` from ``models.get`` - a
derived identity of the SERVED model revision (the API publishes no weights hash). A new
revision therefore becomes a new profile and must be re-materialised; vectors of another
provider/model/revision are never reused.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from app.knowledge.embeddings.errors import (
    EmbeddingCountMismatchError,
    EmbeddingError,
    EmbeddingInputTooLongError,
    EmbeddingModelNotFoundError,
    EmbeddingNotConfiguredError,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
)
from app.knowledge.embeddings.inputs import (
    GEMINI_INPUT_VERSION,
    gemini_document_input,
    gemini_query_input,
)
from app.knowledge.embeddings.profile import normalise_model_tag

logger = logging.getLogger("app.knowledge.embeddings")

MAX_INPUT_CHARS = 16_000  # ~4k tokens of English: well inside the 8 192-token window
MAX_RETRIES_CAP = 2
_PAUSE = 0.5


def classify(exc: BaseException) -> EmbeddingError:
    if isinstance(exc, EmbeddingError):
        return exc
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return EmbeddingTimeoutError()
    if isinstance(exc, httpx.TransportError | ConnectionError):
        return EmbeddingUnavailableError()
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        raw = str(getattr(exc, "message", "") or "").lower()  # classification only
        if code == 400 and (
            "token" in raw and ("exceed" in raw or "long" in raw or "limit" in raw)
        ):
            return EmbeddingInputTooLongError(detail="http_400")
        if code == 404:
            return EmbeddingModelNotFoundError()
        if code in (401, 403):
            return EmbeddingProviderError(detail="auth")
        if code == 429 or code >= 500:
            return EmbeddingUnavailableError(detail=f"http_{code}")
        return EmbeddingProviderError(detail=f"http_{code}")
    return EmbeddingProviderError(detail=type(exc).__name__)


class GeminiEmbeddingProvider:
    provider_name = "gemini"
    input_version = GEMINI_INPUT_VERSION

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        dimensions: int,
        timeout_seconds: float,
        max_retries: int = 1,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model.startswith("gemini-embedding-2"):
            raise ValueError(
                "only the gemini-embedding-2 family (prefix input format) is supported"
            )
        if not 128 <= dimensions <= 3072:
            raise ValueError("gemini-embedding-2 supports 128-3072 dimensions")
        if not 0 <= max_retries <= MAX_RETRIES_CAP:
            raise ValueError(f"max_retries must be between 0 and {MAX_RETRIES_CAP}")
        self._model_id = model
        self._model = normalise_model_tag(model)
        self._api_key = api_key
        self._dimensions = dimensions
        self._timeout_ms = int(timeout_seconds * 1000)
        self._max_retries = max_retries
        self._sleep = sleep
        self._client = client
        self._digest: str | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> GeminiEmbeddingProvider:
        key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
        return cls(
            settings.gemini_embedding_model,
            api_key=key,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries,
        )

    # --- contract ---------------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @staticmethod
    def document_text(title: str, section: str, content: str) -> str:
        return gemini_document_input(title, section, content)

    @staticmethod
    def query_text(query: str) -> str:
        return gemini_query_input(query)

    def resolve_model_digest(self) -> str:
        if self._digest is None:
            model = self._call(lambda: self._sdk().models.get(model=self._model_id))
            name = getattr(model, "name", None) or self._model_id
            version = getattr(model, "version", None) or "unversioned"
            raw = f"gemini:{name}:{version}:{self._dimensions}"
            self._digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self._digest

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text])
        if len(vectors) != 1:
            raise EmbeddingCountMismatchError(detail=f"expected_1_got_{len(vectors)}")
        return vectors[0]

    # --- internals --------------------------------------------------------------------------
    def _sdk(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise EmbeddingNotConfiguredError()
            from google import genai  # noqa: PLC0415
            from google.genai import types  # noqa: PLC0415

            self._client = genai.Client(
                api_key=self._api_key, http_options=types.HttpOptions(timeout=self._timeout_ms)
            )
        return self._client

    def _embed(self, inputs: list[str]) -> list[Any]:
        if any(len(t) > MAX_INPUT_CHARS for t in inputs):
            raise EmbeddingInputTooLongError(detail="provider_guard")  # refused, not truncated
        from google.genai import types  # noqa: PLC0415

        contents = [types.Content(parts=[types.Part(text=t)]) for t in inputs]
        config = types.EmbedContentConfig(output_dimensionality=self._dimensions)
        response = self._call(
            lambda: self._sdk().models.embed_content(
                model=self._model_id, contents=contents, config=config
            )
        )
        embeddings = getattr(response, "embeddings", None)
        if not isinstance(embeddings, list):
            raise EmbeddingCountMismatchError(detail="no_embeddings")
        return [getattr(e, "values", None) for e in embeddings]

    def _call(self, fn: Callable[[], Any]) -> Any:
        attempt = 0
        while True:
            attempt += 1
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - classified into safe errors
                error = classify(exc)
                retry = isinstance(error, EmbeddingTimeoutError | EmbeddingUnavailableError)
                if retry and attempt <= self._max_retries:
                    logger.warning(
                        "embedding call retry",
                        extra={
                            "provider": self.provider_name,
                            "error_code": error.code,
                            "attempt": attempt,
                        },
                    )
                    self._sleep(_PAUSE)
                    continue
                raise error from None
