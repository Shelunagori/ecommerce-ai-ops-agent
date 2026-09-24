"""Validation applied to EVERY vector a provider returns (documents and queries)."""

import math
from collections.abc import Sequence
from typing import Any

from app.knowledge.embeddings.errors import (
    EmbeddingCountMismatchError,
    EmbeddingDimensionMismatchError,
    EmbeddingNonFiniteError,
    EmbeddingZeroVectorError,
)


def validate_vector(vector: Any, dimensions: int) -> list[float]:
    if not isinstance(vector, Sequence) or isinstance(vector, str | bytes):
        raise EmbeddingNonFiniteError(detail="not_a_sequence")
    if len(vector) != dimensions:
        raise EmbeddingDimensionMismatchError(detail=f"expected_{dimensions}_got_{len(vector)}")
    values: list[float] = []
    for v in vector:
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise EmbeddingNonFiniteError()
        values.append(float(v))
    if math.fsum(v * v for v in values) == 0.0:
        raise EmbeddingZeroVectorError()
    return values


def validate_vectors(vectors: Any, expected_count: int, dimensions: int) -> list[list[float]]:
    if not isinstance(vectors, Sequence) or len(vectors) != expected_count:
        got = len(vectors) if isinstance(vectors, Sequence) else "none"
        raise EmbeddingCountMismatchError(detail=f"expected_{expected_count}_got_{got}")
    return [validate_vector(v, dimensions) for v in vectors]
