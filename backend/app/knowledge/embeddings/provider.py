"""Embedding provider contract and the Ollama implementation (Step 8: Ollama only).

Construction never touches the network. Network happens only in:
* ``resolve_model_digest()`` - lists the local Ollama models once and caches the digest of
  the configured tag for the lifetime of the provider object;
* ``embed_documents()`` / ``embed_query()``.

Both embedding methods use ONE request path: the official ``ollama`` client's
``Client.embed(model, input=[...], truncate=False, dimensions=<configured>)``. ``truncate``
is passed explicitly as ``False`` on every request, so an input longer than the model's
context is refused by Ollama (``embedding_input_too_long``) instead of being cut silently
(Ollama's default would truncate, and ``langchain_ollama.OllamaEmbeddings`` does not expose
the switch). Queries are sent as a one-element batch through the same call.

Callers use ``embed_document_inputs`` / ``embed_query_text`` (below), which apply the
versioned input format and validate every returned vector; the provider itself never
prefixes, truncates or post-processes text.

Retry policy (narrow, documented): only timeouts, connection failures and HTTP 5xx from
Ollama are retried, at most ``max_retries`` times (default 1, hard cap 2), with no delay
growth beyond a short fixed pause. Everything else fails immediately with a stable error.
"""

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import httpx

from app.knowledge.embeddings.errors import (
    EmbeddingCountMismatchError,
    EmbeddingError,
    EmbeddingInputTooLongError,
    EmbeddingModelNotFoundError,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
)
from app.knowledge.embeddings.inputs import INPUT_VERSION, query_input
from app.knowledge.embeddings.profile import (
    EmbeddingProfile,
    normalise_digest,
    normalise_model_tag,
)
from app.knowledge.embeddings.validation import validate_vector, validate_vectors

logger = logging.getLogger("app.knowledge.embeddings")

MAX_RETRIES_CAP = 2
_RETRY_PAUSE_SECONDS = 0.5

try:  # the ollama SDK ships with langchain-ollama
    from ollama import ResponseError as OllamaResponseError
except ImportError:  # pragma: no cover
    OllamaResponseError = None  # type: ignore[assignment,misc]


class EmbeddingProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def resolve_model_digest(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def resolve_profile(provider: EmbeddingProvider) -> EmbeddingProfile:
    """The concrete profile of ``provider`` right now (resolves the model digest)."""
    return EmbeddingProfile(
        provider=provider.provider_name,
        model=provider.model_name,
        model_digest=provider.resolve_model_digest(),
        dimensions=provider.dimensions,
        input_version=INPUT_VERSION,
    )


def embed_document_inputs(provider: EmbeddingProvider, inputs: Sequence[str]) -> list[list[float]]:
    """Embed already-formatted document inputs; count, dimension, finiteness and non-zero
    norm are validated for every vector."""
    vectors = provider.embed_documents(list(inputs))
    return validate_vectors(vectors, len(inputs), provider.dimensions)


def embed_query_text(provider: EmbeddingProvider, query: str) -> list[float]:
    """Embed a (cleaned) user query with the versioned query prefix; validated."""
    return validate_vector(provider.embed_query(query_input(query)), provider.dimensions)


def classify(exc: BaseException) -> EmbeddingError:
    if isinstance(exc, EmbeddingError):
        return exc
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return EmbeddingTimeoutError()
    if isinstance(exc, httpx.TransportError | ConnectionError | OSError):
        return EmbeddingUnavailableError()
    if OllamaResponseError is not None and isinstance(exc, OllamaResponseError):
        status = getattr(exc, "status_code", None)
        # The raw server text is only inspected for classification, never surfaced.
        raw = str(getattr(exc, "error", "") or "").lower()
        if "context length" in raw or ("exceed" in raw and "length" in raw):
            return EmbeddingInputTooLongError(detail=f"http_{status}")
        if status == 404:
            return EmbeddingModelNotFoundError()
        if isinstance(status, int) and status >= 500:
            return EmbeddingUnavailableError(detail=f"http_{status}")
        return EmbeddingProviderError(detail=f"http_{status}")
    return EmbeddingProviderError(detail=type(exc).__name__)


def _retryable(error: EmbeddingError) -> bool:
    return isinstance(error, EmbeddingTimeoutError | EmbeddingUnavailableError)


class OllamaEmbeddingProvider:
    """One official ``ollama.Client`` for ``embed`` (vectors) and ``list`` (digest).
    Constructing the client makes no request."""

    provider_name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        dimensions: int,
        timeout_seconds: float,
        max_retries: int = 1,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0 <= max_retries <= MAX_RETRIES_CAP:
            raise ValueError(f"max_retries must be between 0 and {MAX_RETRIES_CAP}")
        self._model = normalise_model_tag(model)
        self._dimensions = dimensions
        self._max_retries = max_retries
        self._sleep = sleep
        self._digest: str | None = None
        if client is None:
            import ollama  # noqa: PLC0415

            # Same base URL as configured; the httpx timeout applies to every request.
            client = ollama.Client(host=base_url, timeout=httpx.Timeout(timeout_seconds))
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def resolve_model_digest(self) -> str:
        """Digest of the configured tag in the local Ollama, resolved once per provider."""
        if self._digest is None:
            listing = self._call(self._client.list)
            models = getattr(listing, "models", None)
            if models is None and isinstance(listing, dict):
                models = listing.get("models", [])
            for entry in models or []:
                name = getattr(entry, "model", None) or (
                    entry.get("model") or entry.get("name") if isinstance(entry, dict) else None
                )
                digest = getattr(entry, "digest", None) or (
                    entry.get("digest") if isinstance(entry, dict) else None
                )
                if isinstance(name, str) and _same_tag(name, self._model) and digest:
                    try:
                        self._digest = normalise_digest(digest)
                    except ValueError:
                        raise EmbeddingProviderError(detail="bad_digest") from None
                    break
            else:
                raise EmbeddingModelNotFoundError(
                    f"The embedding model is not available. Run: ollama pull {self._model}"
                )
        return self._digest

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text])
        if len(vectors) != 1:
            raise EmbeddingCountMismatchError(detail=f"expected_1_got_{len(vectors)}")
        return vectors[0]

    def _embed(self, inputs: list[str]) -> list[Any]:
        """The single embedding request path (documents and queries)."""
        response = self._call(
            lambda: self._client.embed(
                model=self._model,
                input=inputs,
                truncate=False,  # never let Ollama cut an over-long input silently
                dimensions=self._dimensions,
            )
        )
        vectors = getattr(response, "embeddings", None)
        if vectors is None and isinstance(response, dict):
            vectors = response.get("embeddings")
        if not isinstance(vectors, list):
            raise EmbeddingCountMismatchError(detail="no_embeddings")
        return vectors

    def _call(self, fn: Callable[[], Any]) -> Any:
        attempt = 0
        while True:
            attempt += 1
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - classified into safe errors
                error = classify(exc)
                if _retryable(error) and attempt <= self._max_retries:
                    logger.warning(
                        "embedding call retry",
                        extra={
                            "provider": self.provider_name,
                            "model": self._model,
                            "error_code": error.code,
                            "attempt": attempt,
                        },
                    )
                    self._sleep(_RETRY_PAUSE_SECONDS)
                    continue
                raise error from None


def _same_tag(listed: str, configured: str) -> bool:
    try:
        return normalise_model_tag(listed) == configured
    except ValueError:
        return False


def resolve_ollama_model_digest(model: str, *, base_url: str, timeout_seconds: float = 10.0) -> str:
    """Digest of any locally pulled Ollama tag (e.g. the CHAT model, for evaluation
    provenance). Same tag lookup as the embedding profile; no embedding request is made."""
    probe = OllamaEmbeddingProvider(
        model, base_url=base_url, dimensions=1, timeout_seconds=timeout_seconds, max_retries=0
    )
    return probe.resolve_model_digest()


def get_embedding_provider(settings: Any = None) -> OllamaEmbeddingProvider:
    """Build the configured provider from trusted settings (no network)."""
    if settings is None:
        from app.core.config import get_settings  # noqa: PLC0415

        settings = get_settings()
    if settings.embedding_provider != "ollama":  # pragma: no cover - Literal-guarded
        raise ValueError("unsupported embedding provider")
    return OllamaEmbeddingProvider(
        settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
    )
