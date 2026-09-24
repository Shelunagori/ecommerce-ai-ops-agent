"""Hosted Gemini embedding provider (Phase 8) with a fake SDK client: no network, no key."""

import hashlib
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import Settings
from app.knowledge.embeddings import errors as E  # noqa: N812
from app.knowledge.embeddings.gemini import MAX_INPUT_CHARS, GeminiEmbeddingProvider
from app.knowledge.embeddings.inputs import GEMINI_INPUT_VERSION, INPUT_VERSION
from app.knowledge.embeddings.provider import (
    document_text_for,
    embed_document_inputs,
    embed_query_text,
    get_embedding_provider,
    resolve_profile,
)
from tests.knowledge.fakes import hashed_vector


class APIError(Exception):
    def __init__(self, code, message="x"):
        self.code, self.message = code, message
        super().__init__(message)


class FakeModels:
    def __init__(self, dims=768, script=None):
        self.dims = dims
        self.calls = []
        self.gets = 0
        self.script = list(script or [])

    def embed_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, BaseException):
                raise step
        texts = [c.parts[0].text for c in contents]
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=hashed_vector(t, self.dims)) for t in texts]
        )

    def get(self, *, model):
        self.gets += 1
        return SimpleNamespace(name=f"models/{model}", version="2")


def provider(models=None, **kw):
    fake = SimpleNamespace(models=models or FakeModels())
    return GeminiEmbeddingProvider(
        "gemini-embedding-2",
        api_key=None,
        dimensions=768,
        timeout_seconds=5,
        client=fake,
        sleep=lambda _s: None,
        **kw,
    ), fake.models


def test_construction_is_network_free_and_needs_no_key():
    p = GeminiEmbeddingProvider(
        "gemini-embedding-2", api_key=None, dimensions=768, timeout_seconds=5
    )
    assert (p.provider_name, p.model_name, p.dimensions) == (
        "gemini",
        "gemini-embedding-2:latest",
        768,
    )
    with pytest.raises(E.EmbeddingNotConfiguredError):  # only when a request is attempted
        p.embed_query("x")


def test_settings_select_the_hosted_provider():
    s = Settings(_env_file=None, embedding_provider="gemini", embedding_dimensions=768)
    p = get_embedding_provider(s)
    assert isinstance(p, GeminiEmbeddingProvider) and p.model_name == "gemini-embedding-2:latest"


@pytest.mark.parametrize("dims", [64, 4096])
def test_dimension_bounds(dims):
    with pytest.raises(ValueError):
        GeminiEmbeddingProvider(
            "gemini-embedding-2", api_key="k", dimensions=dims, timeout_seconds=5
        )


def test_legacy_model_family_is_refused():
    with pytest.raises(ValueError):
        GeminiEmbeddingProvider(
            "gemini-embedding-001", api_key="k", dimensions=768, timeout_seconds=5
        )


def test_each_text_is_its_own_content_and_dimensions_are_explicit():
    p, models = provider()
    vectors = embed_document_inputs(p, ["a text", "b text", "c text"])
    assert len(vectors) == 3 and all(len(v) == 768 for v in vectors)
    [(model, contents, config)] = models.calls
    assert model == "gemini-embedding-2" and len(contents) == 3
    assert config.output_dimensionality == 768


def test_documented_retrieval_prefixes():
    p, models = provider()
    assert document_text_for(p, "Refund Policy", "Refund timing", "5 days") == (
        "title: Refund Policy | text: Section: Refund timing\n\n5 days"
    )
    embed_query_text(p, "How long is a refund?")
    assert (
        models.calls[0][1][0].parts[0].text == "task: search result | query: How long is a refund?"
    )


def test_profile_is_a_separate_embedding_space():
    p, models = provider()
    profile = resolve_profile(p)
    expected = hashlib.sha256(b"gemini:models/gemini-embedding-2:2:768").hexdigest()
    assert (profile.provider, profile.input_version, profile.model_digest) == (
        "gemini",
        GEMINI_INPUT_VERSION,
        expected,
    )
    assert profile.input_version != INPUT_VERSION
    resolve_profile(p)
    assert models.gets == 1  # resolved once per provider


def test_over_long_input_is_refused_before_any_request():
    p, models = provider()
    with pytest.raises(E.EmbeddingInputTooLongError):
        p.embed_documents(["x" * (MAX_INPUT_CHARS + 1)])
    assert models.calls == []


@pytest.mark.parametrize(
    ("exc", "error", "attempts"),
    [
        (APIError(429), E.EmbeddingUnavailableError, 2),
        (APIError(503), E.EmbeddingUnavailableError, 2),
        (httpx.ReadTimeout("slow"), E.EmbeddingTimeoutError, 2),
        (APIError(404), E.EmbeddingModelNotFoundError, 1),
        (APIError(403, "API key not valid SECRET-XYZ"), E.EmbeddingProviderError, 1),
        (APIError(400, "input token count exceeds the limit"), E.EmbeddingInputTooLongError, 1),
    ],
)
def test_errors_are_classified_and_retried_narrowly(exc, error, attempts):
    models = FakeModels(script=[exc, exc, exc])
    p, _ = provider(models)
    with pytest.raises(error) as got:
        p.embed_query("x")
    assert len(models.calls) == attempts
    # Provider text is used for classification only: never in the message OR the detail
    # (the detail reaches logs and error envelopes).
    assert "secret" not in f"{got.value.message} {got.value.detail}".lower()


def test_returned_vectors_are_validated():
    class Bad(FakeModels):
        def embed_content(self, **kw):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.0] * 768)])

    p, _ = provider(Bad())
    with pytest.raises(E.EmbeddingZeroVectorError):
        embed_query_text(p, "x")
