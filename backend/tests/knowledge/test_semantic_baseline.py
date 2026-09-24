"""Integrity of the committed real-Ollama semantic baseline (data/eval/semantic_baseline_v1.json).

The file was produced on a developer Mac with the real model; it is evidence, not something
CI can regenerate (that needs Ollama: see tests/db/test_embeddings_live.py, opt-in). These
checks keep it honest offline: schema and provenance are exact, fingerprints match the
current fixtures, aggregate metrics follow from the per-case ranks, and the recorded
lexical-vs-semantic comparison is reproducible from the two committed snapshots.
"""

import json
import re
from pathlib import Path

import pytest

from app.core.config import Settings
from app.knowledge.chunking import CHUNKER
from app.knowledge.embeddings.inputs import INPUT_VERSION
from app.knowledge.embeddings.profile import EmbeddingProfile, normalise_model_tag
from app.knowledge.evaluation import DEFAULT_CASES, cases_fingerprint, comparison, load_cases

EVAL_DIR = DEFAULT_CASES.parent
SEMANTIC = EVAL_DIR / "semantic_baseline_v1.json"
LEXICAL = EVAL_DIR / "lexical_baseline_v1.json"
TOP_KEYS = {
    "retriever",
    "cases_file",
    "cases_sha256",
    "chunker",
    "k",
    "embedding_profile",
    "corpus_sha256",
    "metrics",
    "cases",
}
METRIC_KEYS = {"cases", "document_hit@1", "document_hit@3", "chunk_hit@1", "chunk_hit@3"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def snap() -> dict:
    return _load(SEMANTIC)


@pytest.fixture(scope="module")
def lexical() -> dict:
    return _load(LEXICAL)


def _metrics_from_ranks(cases: dict, k: int) -> dict:
    tenant_of = {c.id: c.tenant for c in load_cases()}
    groups: dict[str, list[str]] = {"all": list(cases)}
    for case_id in cases:
        groups.setdefault(tenant_of[case_id], []).append(case_id)

    def rate(ids: list[str], field: str, cutoff: int) -> float:
        hits = sum(1 for i in ids if (r := cases[i][field]) is not None and r <= cutoff)
        return round(hits / len(ids), 3)

    return {
        name: {
            "cases": len(ids),
            "document_hit@1": rate(ids, "document_rank", 1),
            "document_hit@3": rate(ids, "document_rank", k),
            "chunk_hit@1": rate(ids, "chunk_rank", 1),
            "chunk_hit@3": rate(ids, "chunk_rank", k),
        }
        for name, ids in groups.items()
    }


# --- schema and provenance ----------------------------------------------------------------------
def test_schema_is_exact(snap):
    assert set(snap) == TOP_KEYS
    assert snap["retriever"] == "semantic-pgvector-v1"
    assert snap["cases_file"] == "data/eval/policy_retrieval_cases.yaml"
    assert snap["k"] == 3
    assert set(snap["metrics"]) == {"all", "northstar-commerce", "bluepeak-retail"}
    assert all(set(m) == METRIC_KEYS for m in snap["metrics"].values())
    for ranks in snap["cases"].values():
        assert set(ranks) == {"document_rank", "chunk_rank"}
        for r in ranks.values():
            assert r is None or (isinstance(r, int) and not isinstance(r, bool) and 1 <= r <= 3)


def test_provenance_matches_the_configured_profile(snap):
    p = snap["embedding_profile"]
    settings = Settings()
    assert set(p) == {"provider", "model", "model_digest", "dimensions", "input_version", "key"}
    assert snap["chunker"] == CHUNKER == "policy-section-v1"
    assert p["provider"] == settings.embedding_provider == "ollama"
    assert p["model"] == normalise_model_tag(settings.ollama_embedding_model)
    assert p["model"] == "nomic-embed-text-v2-moe:latest"
    assert re.fullmatch(r"[0-9a-f]{64}", p["model_digest"])
    assert p["dimensions"] == settings.embedding_dimensions == 768
    assert p["input_version"] == INPUT_VERSION == "policy-embedding-input-v1"
    rebuilt = EmbeddingProfile(
        p["provider"], p["model"], p["model_digest"], p["dimensions"], p["input_version"]
    )
    assert p["key"] == rebuilt.key and p == rebuilt.as_dict()


def test_cases_fingerprint_and_ids_match_the_current_fixture(snap):
    assert snap["cases_sha256"] == cases_fingerprint()
    assert list(snap["cases"]) == [c.id for c in load_cases()]


def test_no_vectors_or_similarity_scores_are_stored(snap):
    raw = SEMANTIC.read_text(encoding="utf-8")

    def keys(node) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {k for v in node.values() for k in keys(v)}
        if isinstance(node, list):
            return {k for v in node for k in keys(v)}
        return set()

    forbidden = {"score", "similarity", "embedding", "vector", "query", "content", "results"}
    assert keys(snap).isdisjoint(forbidden)

    def floats(node) -> list[float]:
        if isinstance(node, float):
            return [node]
        if isinstance(node, dict):
            return [f for v in node.values() for f in floats(v)]
        if isinstance(node, list):
            return [f for v in node for f in floats(v)]
        return []

    # Only hit rates are floats; everything outside "metrics" is ints/strings/None.
    assert floats({k: v for k, v in snap.items() if k != "metrics"}) == []
    assert len(raw) < 8_000


# --- metrics follow from the ranks ------------------------------------------------------------
def test_aggregate_metrics_agree_with_per_case_ranks(snap):
    assert snap["metrics"] == _metrics_from_ranks(snap["cases"], snap["k"])


def test_lexical_snapshot_metrics_agree_with_its_ranks(lexical):
    assert lexical["metrics"] == _metrics_from_ranks(lexical["cases"], lexical["k"])


def test_recorded_measurement(snap, lexical):
    assert snap["metrics"]["all"] == {
        "cases": 20,
        "document_hit@1": 1.0,
        "document_hit@3": 1.0,
        "chunk_hit@1": 1.0,
        "chunk_hit@3": 1.0,
    }
    assert lexical["metrics"]["all"]["chunk_hit@1"] == 0.4
    assert lexical["metrics"]["all"]["chunk_hit@3"] == 0.85


# --- comparison with the lexical baseline ------------------------------------------------------
def test_comparison_is_12_wins_8_ties_0_losses(snap, lexical):
    assert lexical["k"] == snap["k"] and lexical["chunker"] == snap["chunker"]
    assert list(lexical["cases"]) == list(snap["cases"])
    rows = comparison(lexical["cases"], snap["cases"])
    outcomes = [r["semantic_vs_lexical"] for r in rows]
    assert (outcomes.count("win"), outcomes.count("tie"), outcomes.count("loss")) == (12, 8, 0)


def test_bluepeak_express_priority_vocabulary_mismatch_is_represented(snap, lexical):
    cases = {c.id: c for c in load_cases()}
    express, priority = cases["bp-express-cost"], cases["bp-priority-cost"]
    # Same tenant, same target chunk; BluePeak names its fast option "Priority", so the
    # "express" query shares no key term with the expected chunk.
    assert express.tenant == priority.tenant == "bluepeak-retail"
    assert express.expected == priority.expected == "policy://shipping-policy/v1#chunk-2"
    assert "express" in express.query.lower() and "priority" not in express.query.lower()
    assert lexical["cases"]["bp-express-cost"]["chunk_rank"] is None  # not in lexical top 3
    assert snap["cases"]["bp-express-cost"] == {"document_rank": 1, "chunk_rank": 1}
    row = next(
        r for r in comparison(lexical["cases"], snap["cases"]) if r["case"] == "bp-express-cost"
    )
    assert row["semantic_vs_lexical"] == "win"
