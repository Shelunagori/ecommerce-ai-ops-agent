"""Concrete embedding-profile identity: which embedding SPACE a vector belongs to.

    ollama/nomic-embed-text-v2-moe:latest@<sha256 digest>/768/policy-embedding-input-v1

The resolved model digest is part of the identity because an Ollama tag (``:latest``) is
mutable: a re-pulled build under the same tag is a different vector space. Vectors are only
ever compared within one concrete profile; a new digest is simply a new profile.
"""

import re
from dataclasses import dataclass

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9._/:-]{1,128}$")


def normalise_model_tag(model: str) -> str:
    """``name`` -> ``name:latest`` (explicit tag); ``name:tag`` unchanged."""
    if not isinstance(model, str) or not _NAME.fullmatch(model):
        raise ValueError("invalid model name")
    last = model.rsplit("/", 1)[-1]
    return model if ":" in last else f"{model}:latest"


def normalise_digest(digest: str) -> str:
    value = digest.removeprefix("sha256:") if isinstance(digest, str) else ""
    if not _DIGEST.fullmatch(value):
        raise ValueError("model digest must be a sha256 hex digest")
    return value


@dataclass(frozen=True)
class EmbeddingProfile:
    provider: str
    model: str  # configured model with explicit tag, e.g. "nomic-embed-text-v2-moe:latest"
    model_digest: str  # resolved immutable build (sha256 hex)
    dimensions: int
    input_version: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9-]{1,32}", self.provider):
            raise ValueError("invalid provider")
        if normalise_model_tag(self.model) != self.model:
            raise ValueError("model must carry an explicit tag")
        normalise_digest(self.model_digest)
        if not isinstance(self.dimensions, int) or not 1 <= self.dimensions <= 16000:
            raise ValueError("dimensions must be between 1 and 16000")
        if not re.fullmatch(r"[a-z0-9-]{1,40}", self.input_version):
            raise ValueError("invalid input version")

    @property
    def key(self) -> str:
        return (
            f"{self.provider}/{self.model}@{self.model_digest}/"
            f"{self.dimensions}/{self.input_version}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_digest": self.model_digest,
            "dimensions": self.dimensions,
            "input_version": self.input_version,
            "key": self.key,
        }
