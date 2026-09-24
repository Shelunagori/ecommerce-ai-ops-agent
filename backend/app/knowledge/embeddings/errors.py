"""Stable, safe embedding-domain errors. Messages never contain raw provider responses."""


class EmbeddingError(Exception):
    code = "embedding_error"
    message = "The embedding operation failed."

    def __init__(self, message: str | None = None, *, detail: str | None = None) -> None:
        self.message = message or self.message
        self.detail = detail  # safe internal classification only
        super().__init__(self.message)


class EmbeddingUnavailableError(EmbeddingError):
    code = "embedding_unavailable"
    message = "The embedding provider is unavailable."


class EmbeddingTimeoutError(EmbeddingError):
    code = "embedding_timeout"
    message = "The embedding provider timed out."


class EmbeddingModelNotFoundError(EmbeddingError):
    code = "embedding_model_not_found"
    message = "The configured embedding model is not available from the provider."


class EmbeddingProviderError(EmbeddingError):
    code = "embedding_provider_error"
    message = "The embedding provider returned an error."


class EmbeddingInputTooLongError(EmbeddingError):
    code = "embedding_input_too_long"
    message = (
        "An embedding input exceeds the model's context length; it was refused, not truncated."
    )


class EmbeddingCountMismatchError(EmbeddingError):
    code = "embedding_count_mismatch"
    message = "The embedding provider returned the wrong number of vectors."


class EmbeddingDimensionMismatchError(EmbeddingError):
    code = "embedding_dimension_mismatch"
    message = "An embedding has an unexpected number of dimensions."


class EmbeddingNonFiniteError(EmbeddingError):
    code = "embedding_non_finite"
    message = "An embedding contains NaN, infinity or a non-numeric value."


class EmbeddingZeroVectorError(EmbeddingError):
    code = "embedding_zero_vector"
    message = "An embedding has zero length and cannot be compared by cosine similarity."


class EmbeddingStaleConflictError(EmbeddingError):
    code = "embedding_stale_conflict"
    message = "Stored embeddings for this profile were built from different input."

    def __init__(self, report: object, message: str | None = None) -> None:
        self.report = report
        super().__init__(message)


class EmbeddingProfileNotMaterializedError(EmbeddingError):
    code = "embedding_profile_not_materialized"
    message = (
        "No policy embeddings exist for the current embedding profile. "
        "Run: uv run python -m scripts.embed_policies"
    )


class EmbeddingNotConfiguredError(EmbeddingError):
    code = "embedding_not_configured"
    message = "The hosted embedding provider is not configured (missing API key)."
