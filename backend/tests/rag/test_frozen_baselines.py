"""Step 9 measures orchestration/grounding only: the Step-7/8 retrieval benchmark, its
baselines, chunking and embedding representation stay frozen (byte-identical)."""

import hashlib
from pathlib import Path

import pytest

from app.knowledge.chunking import CHUNKER
from app.knowledge.embeddings.inputs import INPUT_VERSION

EVAL = Path(__file__).resolve().parents[2] / "data" / "eval"
FROZEN = {
    "lexical_baseline_v1.json": None,
    "semantic_baseline_v1.json": "ad61c806755a80b878a164abe0203de7",
    "policy_retrieval_cases.yaml": None,
}


@pytest.mark.parametrize("name", list(FROZEN))
def test_retrieval_artifacts_unchanged(name, frozen_hashes):
    assert hashlib.md5((EVAL / name).read_bytes()).hexdigest() == frozen_hashes[name]  # noqa: S324


@pytest.fixture(scope="module")
def frozen_hashes():
    return {
        "lexical_baseline_v1.json": "6b1fa1ac72001a192992b785f8673b40",
        "semantic_baseline_v1.json": FROZEN["semantic_baseline_v1.json"],
        "policy_retrieval_cases.yaml": "30004e33a0d23635ac2873dda426e2fb",
    }


def test_retrieval_representation_unchanged():
    assert (CHUNKER, INPUT_VERSION) == ("policy-section-v1", "policy-embedding-input-v1")
