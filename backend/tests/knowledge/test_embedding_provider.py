"""Embedding provider contract, profile identity, input format and vector validation.
No database, no network: fake SDK objects are injected into the Ollama provider."""

import math
import socket
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from ollama import ResponseError

from app.core.config import Settings
from app.knowledge.embeddings.errors import (
    EmbeddingCountMismatchError,
    EmbeddingDimensionMismatchError,
    EmbeddingInputTooLongError,
    EmbeddingModelNotFoundError,
    EmbeddingNonFiniteError,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
    EmbeddingZeroVectorError,
)
from app.knowledge.embeddings.inputs import (
    DOCUMENT_PREFIX,
    INPUT_VERSION,
    QUERY_PREFIX,
    clean_query,
    document_input,
    input_hash,
    query_input,
)
from app.knowledge.embeddings.profile import (
    EmbeddingProfile,
    normalise_digest,
    normalise_model_tag,
)
from app.knowledge.embeddings.provider import (
    OllamaEmbeddingProvider,
    embed_document_inputs,
    embed_query_text,
    get_embedding_provider,
    resolve_profile,
)
from app.knowledge.embeddings.validation import validate_vector, validate_vectors
from tests.knowledge.fakes import DIGEST_A, DIGEST_B, BrokenProvider, HashingEmbeddingProvider

D = 768


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("network I/O attempted")

    for target, name in (
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket, "create_connection"),
        (socket, "getaddrinfo"),
    ):
        monkeypatch.setattr(target, name, refuse)
    yield


# --- input representation (policy-embedding-input-v1) ----------------------------------------
def test_document_input_is_exact_and_prefixed():
    text = document_input("Refund Policy", "Refund Policy > Refund timing", "Refunds in 5 days.")
    assert text == (
        "search_document: Title: Refund Policy\n"
        "Section: Refund Policy > Refund timing\n\n"
        "Refunds in 5 days."
    )
    assert INPUT_VERSION == "policy-embedding-input-v1" and DOCUMENT_PREFIX == "search_document: "


def test_query_input_is_exact_and_prefixed():
    assert query_input("express delivery compensation") == (
        "search_query: express delivery compensation"
    )
    assert QUERY_PREFIX == "search_query: "


def test_input_hash_is_the_sha256_of_the_exact_text():
    a = document_input("T", "S", "body")
    assert input_hash(a) == input_hash(document_input("T", "S", "body"))
    assert input_hash(a) != input_hash(document_input("T", "S", "body!"))
    assert input_hash(a) != input_hash(document_input("T2", "S", "body"))
    assert len(input_hash(a)) == 64


def test_query_cleaning_trims_and_caps_but_never_rewrites():
    assert clean_query("  hello  ") == "hello"
    assert clean_query("x" * 900) == "x" * 500
    assert clean_query("   ") == ""
    with pytest.raises(TypeError):
        clean_query(None)  # type: ignore[arg-type]


# --- profile identity ---------------------------------------------------------------------------
def test_profile_key_includes_the_resolved_digest():
    p = EmbeddingProfile("ollama", "nomic-embed-text-v2-moe:latest", DIGEST_A, 768, INPUT_VERSION)
    assert p.key == (
        f"ollama/nomic-embed-text-v2-moe:latest@{DIGEST_A}/768/policy-embedding-input-v1"
    )
    q = EmbeddingProfile("ollama", "nomic-embed-text-v2-moe:latest", DIGEST_B, 768, INPUT_VERSION)
    assert p.key != q.key and p != q


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "nomic-embed-text-v2-moe"},  # tag must be explicit
        {"model_digest": "abc"},
        {"dimensions": 0},
        {"dimensions": 16001},
        {"provider": "Ollama!"},
        {"input_version": "V 1"},
    ],
)
def test_profile_rejects_invalid_parts(kwargs):
    base = {
        "provider": "ollama",
        "model": "m:latest",
        "model_digest": DIGEST_A,
        "dimensions": 768,
        "input_version": INPUT_VERSION,
    }
    with pytest.raises(ValueError):
        EmbeddingProfile(**{**base, **kwargs})


def test_tag_and_digest_normalisation():
    assert normalise_model_tag("nomic-embed-text-v2-moe") == "nomic-embed-text-v2-moe:latest"
    assert normalise_model_tag("ns/model:q4") == "ns/model:q4"
    assert normalise_model_tag("host:1234/model") == "host:1234/model:latest"
    assert normalise_digest(f"sha256:{DIGEST_A}") == DIGEST_A
    for bad in ("", "sha256:xyz", "A" * 64):
        with pytest.raises(ValueError):
            normalise_digest(bad)


# --- vector validation --------------------------------------------------------------------------
def test_valid_vectors_pass_and_are_floats():
    [v] = validate_vectors([[1, 0.5] + [0.0] * (D - 2)], 1, D)
    assert v[:2] == [1.0, 0.5] and all(isinstance(x, float) for x in v)


@pytest.mark.parametrize(
    ("vectors", "count", "error"),
    [
        ([[1.0] * D], 2, EmbeddingCountMismatchError),
        ([[1.0] * D, [1.0] * D], 1, EmbeddingCountMismatchError),
        (None, 1, EmbeddingCountMismatchError),
        ([[1.0] * (D - 1)], 1, EmbeddingDimensionMismatchError),
        ([[1.0] * (D + 1)], 1, EmbeddingDimensionMismatchError),
        ([[math.nan] + [1.0] * (D - 1)], 1, EmbeddingNonFiniteError),
        ([[math.inf] + [1.0] * (D - 1)], 1, EmbeddingNonFiniteError),
        ([[-math.inf] + [1.0] * (D - 1)], 1, EmbeddingNonFiniteError),
        ([["1"] + [1.0] * (D - 1)], 1, EmbeddingNonFiniteError),
        ([[True] + [1.0] * (D - 1)], 1, EmbeddingNonFiniteError),
        (["x" * D], 1, EmbeddingNonFiniteError),
        ([[0.0] * D], 1, EmbeddingZeroVectorError),
    ],
)
def test_invalid_vectors_are_rejected(vectors, count, error):
    with pytest.raises(error):
        validate_vectors(vectors, count, D)


def test_tiny_but_nonzero_vector_is_accepted():
    v = [0.0] * D
    v[5] = 1e-30
    assert validate_vector(v, D)[5] == 1e-30


# --- helper functions every caller uses -----------------------------------------------------------
def test_document_batch_goes_through_validation():
    fake = HashingEmbeddingProvider()
    inputs = [document_input("T", "S", f"body {i}") for i in range(3)]
    vectors = embed_document_inputs(fake, inputs)
    assert len(vectors) == 3 and fake.document_calls == [inputs]


def test_query_embedding_adds_the_query_prefix():
    fake = HashingEmbeddingProvider()
    embed_query_text(fake, "express delivery")
    assert fake.query_calls == ["search_query: express delivery"]


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ([[0.0] * D], EmbeddingZeroVectorError),
        ([[math.nan] * D], EmbeddingNonFiniteError),
        ([[1.0] * 256], EmbeddingDimensionMismatchError),
        ([], EmbeddingCountMismatchError),
    ],
)
def test_bad_document_vectors_from_a_provider(result, error):
    with pytest.raises(error):
        embed_document_inputs(BrokenProvider(documents_result=result), ["search_document: x"])


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ([0.0] * D, EmbeddingZeroVectorError),
        ([math.inf] * D, EmbeddingNonFiniteError),
        ([1.0] * 10, EmbeddingDimensionMismatchError),
        (None, EmbeddingNonFiniteError),
    ],
)
def test_bad_query_vectors_from_a_provider(result, error):
    with pytest.raises(error):
        embed_query_text(BrokenProvider(query_result=result), "refund")


def test_resolve_profile_uses_the_provider_digest():
    fake = HashingEmbeddingProvider(digest=DIGEST_B)
    profile = resolve_profile(fake)
    assert (profile.model_digest, profile.input_version, profile.dimensions) == (
        DIGEST_B,
        INPUT_VERSION,
        768,
    )


# --- Ollama provider (injected fake ollama.Client) ---------------------------------------
class FakeClient:
    """Stands in for ``ollama.Client``: records every ``embed``/``list`` call verbatim."""

    def __init__(self, models: list[dict[str, str]] | BaseException = (), script=None) -> None:
        self.models = models
        self.script = list(script or [])
        self.list_calls = 0
        self.embed_calls: list[dict[str, Any]] = []

    def list(self) -> Any:
        self.list_calls += 1
        if isinstance(self.models, BaseException):
            raise self.models
        return {"models": list(self.models)}

    def embed(self, **kwargs: Any) -> Any:
        self.embed_calls.append(kwargs)
        step = self.script.pop(0) if self.script else None
        if isinstance(step, BaseException):
            raise step
        if step is None:  # default: one valid vector per input
            return {"embeddings": [[1.0] + [0.0] * (D - 1) for _ in kwargs["input"]]}
        return step


REQ = httpx.Request("POST", "http://localhost:11434/api/embed")


def make(client=None, retries=1, model="nomic-embed-text-v2-moe", dimensions=D):
    return OllamaEmbeddingProvider(
        model,
        base_url="http://localhost:11434",
        dimensions=dimensions,
        timeout_seconds=5,
        max_retries=retries,
        client=client or FakeClient(),
        sleep=lambda _s: None,
    )


def test_real_construction_makes_no_network_calls(no_network):
    provider = get_embedding_provider(Settings())
    assert (provider.provider_name, provider.model_name, provider.dimensions) == (
        "ollama",
        "nomic-embed-text-v2-moe:latest",
        768,
    )
    OllamaEmbeddingProvider("m", base_url="http://x:1", dimensions=8, timeout_seconds=1)


def test_real_client_keeps_base_url_and_timeout():
    provider = OllamaEmbeddingProvider(
        "m", base_url="http://ollama.internal:11434", dimensions=8, timeout_seconds=7
    )
    http = provider._client._client  # the official client's underlying httpx.Client
    assert str(http.base_url).rstrip("/") == "http://ollama.internal:11434"
    assert http.timeout.read == 7 and http.timeout.connect == 7


def test_settings_drive_the_provider():
    s = Settings(ollama_embedding_model="other-embed:v2", embedding_dimensions=256)
    provider = get_embedding_provider(s)
    assert (provider.model_name, provider.dimensions) == ("other-embed:v2", 256)


def test_documents_use_embed_with_truncate_false_and_dimensions_in_one_batch():
    client = FakeClient()
    batch = ["search_document: a", "search_document: b", "search_document: c"]
    vectors = make(client).embed_documents(batch)
    assert len(vectors) == 3
    assert client.embed_calls == [
        {
            "model": "nomic-embed-text-v2-moe:latest",
            "input": batch,
            "truncate": False,
            "dimensions": 768,
        }
    ]


def test_query_uses_the_same_embed_path_with_truncate_false():
    client = FakeClient()
    vector = make(client).embed_query("search_query: express delivery")
    assert len(vector) == 768
    assert client.embed_calls == [
        {
            "model": "nomic-embed-text-v2-moe:latest",
            "input": ["search_query: express delivery"],
            "truncate": False,
            "dimensions": 768,
        }
    ]


def test_configured_dimensions_are_passed():
    client = FakeClient(script=[{"embeddings": [[1.0] * 256]}])
    make(client, dimensions=256).embed_query("q")
    assert client.embed_calls[0]["dimensions"] == 256 and client.embed_calls[0]["truncate"] is False


def test_materialization_batches_stay_batched_through_the_client():
    client = FakeClient()
    provider = make(client)
    inputs = [document_input("T", "S", f"body {i}") for i in range(5)]
    embed_document_inputs(provider, inputs)
    assert [c["input"] for c in client.embed_calls] == [inputs]  # one request, 5 inputs


def test_text_is_passed_through_unchanged():
    client = FakeClient()
    long_text = "search_document: " + "word " * 3000  # far beyond 512 tokens
    make(client).embed_documents([long_text])
    assert client.embed_calls[0]["input"] == [long_text]  # never cut in application code


@pytest.mark.parametrize(
    "message",
    [
        "the input length exceeds the context length",
        "input length exceeds maximum context length",
    ],
)
def test_over_limit_input_is_surfaced_not_accepted(message):
    client = FakeClient(script=[ResponseError(message + " SECRET-DETAIL", 400)])
    with pytest.raises(EmbeddingInputTooLongError) as caught:
        make(client).embed_documents(["search_document: " + "x " * 5000])
    assert caught.value.code == "embedding_input_too_long"
    assert "SECRET-DETAIL" not in str(caught.value) and len(client.embed_calls) == 1  # no retry


def test_query_over_limit_is_surfaced_too():
    client = FakeClient(script=[ResponseError("input exceeds the context length", 500)])
    with pytest.raises(EmbeddingInputTooLongError):
        make(client).embed_query("search_query: long")
    assert len(client.embed_calls) == 1  # not treated as a retryable 5xx


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ({"embeddings": []}, EmbeddingCountMismatchError),
        ({"embeddings": [[1.0] * D, [1.0] * D]}, EmbeddingCountMismatchError),
        ({"model": "x"}, EmbeddingCountMismatchError),
    ],
)
def test_malformed_query_responses(response, error):
    with pytest.raises(error):
        make(FakeClient(script=[response])).embed_query("q")


def test_digest_is_resolved_once_and_cached():
    client = FakeClient(
        [
            {"model": "llama3.2:3b", "digest": DIGEST_B},
            {"model": "nomic-embed-text-v2-moe:latest", "digest": f"sha256:{DIGEST_A}"},
        ]
    )
    provider = make(client)
    assert client.list_calls == 0  # nothing resolved at construction
    assert provider.resolve_model_digest() == DIGEST_A
    assert provider.resolve_model_digest() == DIGEST_A
    assert client.list_calls == 1


def test_missing_model_is_a_clear_error():
    provider = make(FakeClient([{"model": "llama3.2:3b", "digest": DIGEST_B}]))
    with pytest.raises(EmbeddingModelNotFoundError, match="ollama pull nomic-embed-text-v2-moe"):
        provider.resolve_model_digest()


def test_malformed_digest_is_rejected():
    provider = make(
        FakeClient([{"model": "nomic-embed-text-v2-moe:latest", "digest": "not-a-digest"}])
    )
    with pytest.raises(EmbeddingProviderError):
        provider.resolve_model_digest()


def test_timeout_is_retried_once_then_succeeds():
    client = FakeClient(script=[httpx.ReadTimeout("t", request=REQ)])
    assert len(make(client, retries=1).embed_documents(["x"])) == 1
    assert len(client.embed_calls) == 2
    assert client.embed_calls[0] == client.embed_calls[1]  # retried with truncate=False too


@pytest.mark.parametrize(
    ("exc", "error", "attempts"),
    [
        (httpx.ReadTimeout("t", request=REQ), EmbeddingTimeoutError, 2),
        (httpx.ConnectError("refused", request=REQ), EmbeddingUnavailableError, 2),
        (ResponseError("server exploded SECRET-DETAIL", 500), EmbeddingUnavailableError, 2),
        (ResponseError("model 'x' not found SECRET-DETAIL", 404), EmbeddingModelNotFoundError, 1),
        (ResponseError("bad request SECRET-DETAIL", 400), EmbeddingProviderError, 1),
        (ValueError("internal SECRET-DETAIL"), EmbeddingProviderError, 1),
    ],
)
def test_errors_are_classified_bounded_and_safe(exc, error, attempts):
    client = FakeClient(script=[exc, exc, exc, exc])
    with pytest.raises(error) as caught:
        make(client, retries=1).embed_query("q")
    assert len(client.embed_calls) == attempts
    assert "SECRET-DETAIL" not in str(caught.value) and caught.value.__cause__ is None


def test_retries_are_capped():
    with pytest.raises(ValueError):
        make(retries=3)
    client = FakeClient(script=[httpx.ReadTimeout("t", request=REQ)] * 5)
    with pytest.raises(EmbeddingTimeoutError):
        make(client, retries=0).embed_query("q")
    assert len(client.embed_calls) == 1


def test_unreachable_server_during_digest_resolution():
    provider = make(FakeClient(httpx.ConnectError("refused", request=REQ)))
    with pytest.raises(EmbeddingUnavailableError):
        provider.resolve_model_digest()
