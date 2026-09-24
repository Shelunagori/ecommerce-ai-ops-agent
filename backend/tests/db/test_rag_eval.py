"""RAG/agent evaluation fixture on real PostgreSQL + pgvector with scripted models.

The ORACLE suite shows the evaluator and pipeline score correct behaviour at 1.00; the
ADVERSARIAL suite shows every grounding violation is rejected with its stable error. Live
model numbers come from scripts/eval_rag.py (opt-in, Ollama) and are a measurement, not CI.
"""

import json
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import sessionmaker

import app.db.session as db_session_module
from app.agent.rag.evaluation import (
    DEFAULT_RAG_CASES,
    METRIC_NAMES,
    load_rag_cases,
    rag_cases_fingerprint,
    rag_metrics,
    run_rag_case,
    run_rag_cases,
)
from app.agent.tools import ToolDependencies, build_commerce_tools
from app.knowledge.chunking import ChunkingConfig
from app.knowledge.embeddings.materialize import materialize_embeddings
from app.knowledge.evaluation import DEFAULT_CASES
from app.knowledge.ingest import ingest_policies
from app.knowledge.semantic import SemanticKnowledgeRetriever
from app.knowledge.sources import DEFAULT_POLICY_DIR
from scripts.seed_demo import BLUEPEAK, NORTHSTAR, tenant_id_for
from tests.assistant.fakes import make_provider
from tests.db.conftest import fixed_clock
from tests.knowledge.fakes import HashingEmbeddingProvider
from tests.rag.eval_scripts import ADVERSARIES, oracle_script

TENANTS = {t.slug: tenant_id_for(t.slug) for t in (NORTHSTAR, BLUEPEAK)}


@pytest.fixture
def env(db_session, db_engine, monkeypatch):
    ingest_policies(db_session, DEFAULT_POLICY_DIR, ChunkingConfig())
    provider = HashingEmbeddingProvider()
    materialize_embeddings(db_session, provider, batch_size=16)
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    tools = build_commerce_tools(
        ToolDependencies(session_scope=db_session_module.read_only_session, clock=fixed_clock)
    )

    @contextmanager
    def scope():
        yield db_session

    def retriever_for(day):
        noon = datetime(day.year, day.month, day.day, 12, tzinfo=UTC)
        return SemanticKnowledgeRetriever(scope, provider, clock=lambda: noon)

    return tools, retriever_for


def oracle_provider(case):
    return make_provider(*oracle_script(case))[0]


def test_fixture_is_separate_from_the_frozen_retrieval_benchmark():
    assert DEFAULT_RAG_CASES != DEFAULT_CASES
    assert DEFAULT_RAG_CASES.name == "rag_agent_cases.yaml"
    assert len(rag_cases_fingerprint()) == 64


def test_every_case_has_an_explicit_evaluation_date():
    cases = load_rag_cases()
    assert all(c.as_of is not None for c in cases)
    ids = {c.id for c in cases}
    for required in (
        "bp-delay-current",
        "bp-delay-historical",
        "ns-refund-current",
        "bp-return-window",
        "ns-cancellation",
        "ns-no-applicable-policy",
        "ns-order-commerce-only",
        "ns-mixed-shipment-policy",
        "bp-followup-fresh-retrieval",
    ):
        assert required in ids
    groups = [c.isolation_group for c in cases if c.isolation_group]
    assert groups.count("delay-same-question") == 2
    same = [c for c in cases if c.isolation_group == "delay-same-question"]
    assert same[0].turns == same[1].turns and same[0].tenant != same[1].tenant


def test_oracle_suite_scores_one_on_every_metric(env):
    tools, retriever_for = env
    outcomes = run_rag_cases(load_rag_cases(), oracle_provider, retriever_for, tools, TENANTS)
    failed = [o for o in outcomes if not o["passed"]]
    assert failed == []
    metrics = rag_metrics(outcomes)
    assert set(metrics) == {*METRIC_NAMES, "cases"}
    assert all(metrics[m] == 1.0 for m in METRIC_NAMES), metrics
    by_id = {o["case"]: o for o in outcomes}
    assert by_id["ns-order-commerce-only"]["retrieval_performed"] is False
    assert by_id["ns-no-applicable-policy"]["no_result_abstained"] is True
    assert by_id["ns-mixed-shipment-policy"]["commerce_tools"] == ["get_shipment"]
    # outcomes are structured, never prose
    assert "answer" not in json.dumps(outcomes)


@pytest.mark.parametrize(
    ("name", "case_id", "script", "expected"), ADVERSARIES, ids=[a[0] for a in ADVERSARIES]
)
def test_adversarial_models_are_rejected(env, name, case_id, script, expected):
    tools, retriever_for = env
    case = next(c for c in load_rag_cases() if c.id == case_id)
    provider = make_provider(*script(case))[0]
    outcome = run_rag_case(case, provider, retriever_for, tools, TENANTS[case.tenant])
    assert (outcome["error_code"], outcome["error_detail"]) == expected
    assert outcome["citations"] == []  # nothing ungrounded reached the result


def test_metrics_detect_a_wrong_trajectory(env):
    """The evaluator is not vacuous: a model that skips retrieval fails the decision metric."""
    tools, retriever_for = env
    case = next(c for c in load_rag_cases() if c.id == "bp-return-window")
    lazy = make_provider(*[m for m in oracle_script(load_rag_cases()[-1])])[0]  # greeting answer
    outcome = run_rag_case(case, lazy, retriever_for, tools, TENANTS[case.tenant])
    assert outcome["retrieval_decision_correct"] is False and outcome["passed"] is False
    assert rag_metrics([outcome])["retrieval_decision_accuracy"] == 0.0


def test_live_measurement_cli_writes_provenance_and_no_prose(
    env, db_session, monkeypatch, tmp_path, capsys
):
    """scripts/eval_rag.py end to end with a scripted model (the Mac run uses Ollama)."""
    from scripts import eval_rag

    tools, retriever_for = env
    cases = load_rag_cases()
    provider = make_provider(*[s for c in cases for s in oracle_script(c)])[0]

    @contextmanager
    def scope():
        yield db_session

    monkeypatch.setattr(eval_rag, "get_llm_provider", lambda **_k: provider)
    monkeypatch.setattr(eval_rag, "get_embedding_provider", lambda: HashingEmbeddingProvider())
    monkeypatch.setattr(eval_rag, "read_only_session", scope)
    monkeypatch.setattr(eval_rag, "build_commerce_tools", lambda: tools)
    target = tmp_path / "rag_live_measurement_v1.json"
    assert eval_rag.main(["--provider", "ollama", "--write", str(target)]) == 0
    capsys.readouterr()
    m = json.loads(target.read_text())
    assert m["kind"] == "rag-live-measurement" and m["prompt_version"] == "commerce-assistant-v2"
    assert m["cases_sha256"] == rag_cases_fingerprint() and len(m["corpus_sha256"]) == 64
    assert set(m["llm"]) == {"provider", "model", "model_digest"}
    assert m["embedding_profile"]["input_version"] == "policy-embedding-input-v1"
    assert datetime.fromisoformat(m["generated_at"]).tzinfo is not None
    assert m["metrics"]["case_pass_rate"] == 1.0 and len(m["cases"]) == len(cases)
    raw = target.read_text()
    for prose in ("According to the policy", "Here is the requested", "content", "query"):
        assert prose not in raw
