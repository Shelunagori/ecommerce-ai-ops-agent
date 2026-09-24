"""Lexical retrieval baseline on the evaluation cases (real PostgreSQL, real corpus).

The recorded numbers in ``data/eval/lexical_baseline_v1.json`` are the Step-7 reference
for Step 8. The test fails if the lexical baseline drifts, so any change to the retriever,
chunker or corpus must update the recorded baseline deliberately (never tuned to 100%).
"""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from app.knowledge.chunking import ChunkingConfig
from app.knowledge.evaluation import load_cases, metrics, run_cases
from app.knowledge.ingest import ingest_policies
from app.knowledge.retrieval import LexicalPolicyRetriever
from app.knowledge.sources import DEFAULT_POLICY_DIR
from scripts.seed_demo import BLUEPEAK, NORTHSTAR, tenant_id_for

BASELINE = Path(__file__).resolve().parents[2] / "data" / "eval" / "lexical_baseline_v1.json"
TENANT_IDS = {t.slug: tenant_id_for(t.slug) for t in (NORTHSTAR, BLUEPEAK)}


@pytest.fixture
def outcomes(db_session):
    ingest_policies(db_session, DEFAULT_POLICY_DIR, ChunkingConfig())

    @contextmanager
    def scope():
        yield db_session

    return run_cases(LexicalPolicyRetriever(scope), load_cases(), TENANT_IDS, k=3)


def test_cases_file_is_valid_and_covers_both_tenants():
    cases = load_cases()
    assert len(cases) == 20
    assert {c.tenant for c in cases} == set(TENANT_IDS)
    assert {c.as_of.isoformat() for c in cases} >= {"2026-06-10", "2026-09-01"}


def test_lexical_baseline_matches_the_recorded_reference(outcomes):
    recorded = json.loads(BASELINE.read_text())
    assert recorded["retriever"] == "lexical-pg-fts-v1"
    assert recorded["chunker"] == "policy-section-v1"
    assert metrics(outcomes) == recorded["metrics"]
    per_case = {
        o.case.id: {"document_rank": o.document_rank(), "chunk_rank": o.chunk_rank()}
        for o in outcomes
    }
    assert per_case == recorded["cases"]


def test_every_expected_document_is_in_the_top_three(outcomes):
    """The one property the baseline must keep: right policy document AND version."""
    misses = [o.case.id for o in outcomes if o.document_rank() is None]
    assert misses == []
