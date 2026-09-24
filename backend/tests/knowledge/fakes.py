"""Deterministic fake embedding providers (no network, no model).

* ``HashingEmbeddingProvider``: feature-hashed bag of crudely stemmed words. Texts sharing
  words get high cosine similarity - enough to exercise ranking end to end, NOT a model.
* ``TableEmbeddingProvider``: exact vectors per input text (for cosine-order tests).
* Both record every call (inputs, digest resolutions) so tests can assert prefixes,
  batching and that query vectors are computed but never stored.
"""

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
_WORD = re.compile(r"[a-z0-9]+")
_IGNORED = frozenset(
    "search_document search_query search document query title section the a an of to and or "
    "is are be can i my for in on it what how when which who do does".split()
)


def _stem(word: str) -> str:
    for suffix in ("ation", "ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def hashed_vector(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for word in _WORD.findall(text.lower().replace("search_", "search_ ")):
        if word in _IGNORED:
            continue
        h = hashlib.sha256(_stem(word).encode()).digest()
        index = int.from_bytes(h[:4], "big") % dimensions
        vector[index] += 1.0 if h[4] & 1 else -1.0
    if not any(vector):
        vector[0] = 1.0
    return vector


@dataclass
class _Recorder:
    provider_name: str = "ollama"
    model_name: str = "nomic-embed-text-v2-moe:latest"
    dimensions: int = 768
    digest: str = DIGEST_A
    document_calls: list[list[str]] = field(default_factory=list)
    query_calls: list[str] = field(default_factory=list)
    digest_calls: int = 0

    def resolve_model_digest(self) -> str:
        self.digest_calls += 1
        return self.digest


@dataclass
class HashingEmbeddingProvider(_Recorder):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [hashed_vector(t, self.dimensions) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return hashed_vector(text, self.dimensions)


@dataclass
class TableEmbeddingProvider(_Recorder):
    """Returns ``table[text]``; unknown documents get ``default`` (or fail)."""

    table: dict[str, list[float]] = field(default_factory=dict)
    default: Callable[[str], list[float]] | None = None

    def _lookup(self, text: str) -> list[float]:
        if text in self.table:
            return self.table[text]
        if self.default is not None:
            return self.default(text)
        raise KeyError(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._lookup(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._lookup(text)


@dataclass
class BrokenProvider(_Recorder):
    """Returns a fixed (malformed) payload or raises ``error`` for every call."""

    documents_result: Any = None
    query_result: Any = None
    error: BaseException | None = None

    def embed_documents(self, texts: Sequence[str]) -> Any:
        self.document_calls.append(list(texts))
        if self.error:
            raise self.error
        return (
            self.documents_result(texts)
            if callable(self.documents_result)
            else (self.documents_result)
        )

    def embed_query(self, text: str) -> Any:
        self.query_calls.append(text)
        if self.error:
            raise self.error
        return self.query_result


def unit(dimensions: int, index: int, other: int | None = None, weight: float = 0.0) -> list[float]:
    v = [0.0] * dimensions
    v[index] = 1.0
    if other is not None:
        v[other] = weight
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]
