"""embed_policies / search_policies --retriever semantic / eval_retrieval CLIs, and the
retriever-agnostic evaluation pipeline, with a deterministic fake embedding provider."""

import json
from contextlib import contextmanager

import pytest

import app.db.session as db_session_module
from app.core.config import get_settings
from app.knowledge.chunking import ChunkingConfig
from app.knowledge.embeddings.materialize import materialize_embeddings
from app.knowledge.evaluation import (
    cases_fingerprint,
    comparison,
    corpus_fingerprint,
    load_cases,
    metrics,
    per_case,
    run_cases,
)
from app.knowledge.ingest import ingest_policies
from app.knowledge.semantic import SemanticKnowledgeRetriever
from app.knowledge.sources import DEFAULT_POLICY_DIR
from scripts import embed_policies as embed_cli
from scripts import eval_retrieval as eval_cli
from scripts import search_policies as search_cli
from scripts.seed_demo import BLUEPEAK, NORTHSTAR, tenant_id_for
from tests.knowledge.fakes import DIGEST_A, DIGEST_B, HashingEmbeddingProvider

TENANTS = {t.slug: tenant_id_for(t.slug) for t in (NORTHSTAR, BLUEPEAK)}


@pytest.fixture
def kb(db_session):
    ingest_policies(db_session, DEFAULT_POLICY_DIR, ChunkingConfig())
    return db_session


@pytest.fixture
def wired(kb, monkeypatch):
    @contextmanager
    def scope():
        yield kb

    fake = HashingEmbeddingProvider()
    monkeypatch.setattr(db_session_module, "unit_of_work", scope)
    for module in (search_cli, eval_cli):
        monkeypatch.setattr(module, "read_only_session", scope)
    for module in (embed_cli, search_cli, eval_cli):
        monkeypatch.setattr(module, "get_embedding_provider", lambda *_a, **_k: fake)
    return fake


# --- embed_policies ----------------------------------------------------------------------------
def test_embed_cli_twice_is_idempotent(wired, capsys):
    assert embed_cli.main([]) == 0
    first = json.loads(capsys.readouterr().out)
    assert (first["status"], first["embedded"], first["unchanged"], first["total_chunks"]) == (
        "ok",
        51,
        0,
        51,
    )
    assert first["profile"]["model_digest"] == DIGEST_A and first["dimensions"] == 768
    assert embed_cli.main([]) == 0
    second = json.loads(capsys.readouterr().out)
    assert (second["embedded"], second["unchanged"], second["batches"]) == (0, 51, 0)


def test_embed_cli_refuses_production(monkeypatch, capsys):
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    try:
        assert embed_cli.main([]) == 2
    finally:
        get_settings.cache_clear()
    assert "Refusing" in capsys.readouterr().err


# --- search_policies --retriever semantic ---------------------------------------------------------
def test_semantic_search_cli_both_tenants(wired, kb, capsys):
    materialize_embeddings(kb, wired, batch_size=16)
    for tenant in TENANTS.values():
        code = search_cli.main(
            [
                "--retriever",
                "semantic",
                "--tenant",
                str(tenant),
                "--query",
                "express delivery compensation",
                "--as-of",
                "2026-09-01",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        data = json.loads(out)
        assert data["retriever"] == "semantic-pgvector-v1" and data["score_type"] == (
            "cosine_similarity"
        )
        assert DIGEST_A in data["embedding_profile"] and len(data["results"]) == 5
        assert str(tenant) not in out and ".md" not in out and "[0." not in out


def test_semantic_search_cli_historical(wired, kb, capsys):
    materialize_embeddings(kb, wired, batch_size=16)
    assert (
        search_cli.main(
            [
                "--retriever",
                "semantic",
                "--tenant",
                str(TENANTS["bluepeak-retail"]),
                "--query",
                "delayed shipment compensation",
                "--as-of",
                "2026-06-10",
                "--limit",
                "10",
            ]
        )
        == 0
    )
    results = json.loads(capsys.readouterr().out)["results"]
    assert {
        r["version"] for r in results if r["document_key"] == "delayed-shipment-compensation"
    } == {1}


def test_semantic_search_cli_reports_unmaterialized_profile(wired, capsys):
    assert (
        search_cli.main(
            [
                "--retriever",
                "semantic",
                "--tenant",
                str(TENANTS["northstar-commerce"]),
                "--query",
                "refund",
            ]
        )
        == 1
    )
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert err["error"]["code"] == "embedding_profile_not_materialized"


def test_lexical_remains_the_default_cli_retriever(wired, capsys):
    assert (
        search_cli.main(["--tenant", str(TENANTS["northstar-commerce"]), "--query", "PO boxes"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["retriever"] == "lexical-pg-fts-v1"


# --- evaluation ------------------------------------------------------------------------------
def test_semantic_evaluation_pipeline_runs_on_the_same_cases(wired, kb):
    materialize_embeddings(kb, wired, batch_size=16)

    @contextmanager
    def scope():
        yield kb

    outcomes = run_cases(SemanticKnowledgeRetriever(scope, wired), load_cases(), TENANTS, k=3)
    assert len(outcomes) == 20 and all(len(o.citations) == 3 for o in outcomes)
    m = metrics(outcomes)
    assert set(m) == {"all", "northstar-commerce", "bluepeak-retail"}
    assert set(m["all"]) == {
        "cases",
        "document_hit@1",
        "document_hit@3",
        "chunk_hit@1",
        "chunk_hit@3",
    }


def test_comparison_classifies_wins_losses_and_ties():
    base = {
        "a": {"chunk_rank": 2, "document_rank": 1},
        "b": {"chunk_rank": 1, "document_rank": 1},
        "c": {"chunk_rank": None, "document_rank": 2},
        "d": {"chunk_rank": 3, "document_rank": 1},
    }
    cand = {
        "a": {"chunk_rank": 1, "document_rank": 1},
        "b": {"chunk_rank": 3, "document_rank": 1},
        "c": {"chunk_rank": 2, "document_rank": 1},
        "d": {"chunk_rank": 3, "document_rank": 1},
    }
    assert [r["semantic_vs_lexical"] for r in comparison(base, cand)] == [
        "win",
        "loss",
        "win",
        "tie",
    ]


def test_eval_cli_writes_a_snapshot_with_provenance_and_no_vectors(wired, kb, tmp_path, capsys):
    materialize_embeddings(kb, wired, batch_size=16)
    target = tmp_path / "semantic_baseline_v1.json"
    assert eval_cli.main(["--retriever", "semantic", "--write", str(target)]) == 0
    report = json.loads(capsys.readouterr().out)
    snap = json.loads(target.read_text())
    assert snap["retriever"] == "semantic-pgvector-v1"
    assert snap["embedding_profile"] == {
        "provider": "ollama",
        "model": "nomic-embed-text-v2-moe:latest",
        "model_digest": DIGEST_A,
        "dimensions": 768,
        "input_version": "policy-embedding-input-v1",
        "key": f"ollama/nomic-embed-text-v2-moe:latest@{DIGEST_A}/768/policy-embedding-input-v1",
    }
    assert snap["cases_sha256"] == cases_fingerprint() and snap["k"] == 3
    assert snap["corpus_sha256"] == corpus_fingerprint(kb)
    assert set(snap["cases"]) == {c.id for c in load_cases()}
    assert set(snap["metrics"]) == {"all", "northstar-commerce", "bluepeak-retail"}
    raw = target.read_text()
    assert "score" not in raw and 'embedding":' not in raw.replace("embedding_profile", "")
    assert len(report["comparison_vs_lexical"]) == 20 and "lexical_metrics" in report


def test_eval_cli_lexical_matches_the_recorded_lexical_baseline(wired, kb, capsys):
    assert eval_cli.main(["--retriever", "lexical"]) == 0
    report = json.loads(capsys.readouterr().out)
    recorded = json.loads(
        (DEFAULT_POLICY_DIR.parents[0] / "eval" / "lexical_baseline_v1.json").read_text()
    )
    assert report["metrics"] == recorded["metrics"] and report["cases"] == recorded["cases"]


def test_eval_cli_fails_cleanly_for_an_unmaterialized_profile(wired, capsys, monkeypatch):
    other = HashingEmbeddingProvider(digest=DIGEST_B)
    monkeypatch.setattr(eval_cli, "get_embedding_provider", lambda *_a, **_k: other)
    assert eval_cli.main(["--retriever", "semantic"]) == 1
    assert "embedding_profile_not_materialized" in capsys.readouterr().err


def test_per_case_shape(wired, kb):
    @contextmanager
    def scope():
        yield kb

    from app.knowledge.retrieval import LexicalPolicyRetriever

    rows = per_case(run_cases(LexicalPolicyRetriever(scope), load_cases(), TENANTS))
    assert all(set(v) == {"document_rank", "chunk_rank"} for v in rows.values())
