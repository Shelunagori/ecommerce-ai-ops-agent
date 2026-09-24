"""OPT-IN live RAG tests: real chat model + nomic-embed-text-v2-moe + PostgreSQL/pgvector.

    ollama pull qwen3:4b-instruct && ollama pull nomic-embed-text-v2-moe
    RUN_OLLAMA_INTEGRATION=1 [LIVE_RAG_MODEL=qwen3:4b-instruct] TEST_DATABASE_URL=... \\
        uv run pytest tests/db/test_rag_live.py

Model: ``LIVE_RAG_MODEL`` when set, else the configured ``OLLAMA_MODEL``. Checks behaviour
classes, never prose: the model requests retrieval, the semantic retriever runs, the right
document version is retrieved, the answer carries only current-run citations; a pure
commerce question does not retrieve. If the small model does not extract a historical date,
the test reports it (xfail) instead of forcing the expected call.
"""

import os
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import sessionmaker

import app.db.session as db_session_module
from app.agent.llm import get_llm_provider
from app.agent.rag.evaluation import load_rag_cases, run_rag_case
from app.agent.tools import ToolDependencies, build_commerce_tools
from app.knowledge.chunking import ChunkingConfig
from app.knowledge.embeddings.materialize import materialize_embeddings
from app.knowledge.embeddings.provider import get_embedding_provider
from app.knowledge.ingest import ingest_policies
from app.knowledge.semantic import SemanticKnowledgeRetriever
from app.knowledge.sources import DEFAULT_POLICY_DIR
from scripts.seed_demo import tenant_id_for

pytestmark = [
    pytest.mark.llm_integration,
    pytest.mark.skipif(
        os.getenv("RUN_OLLAMA_INTEGRATION") != "1", reason="set RUN_OLLAMA_INTEGRATION=1"
    ),
]
CASES = {c.id: c for c in load_rag_cases()}


@pytest.fixture(scope="module")
def embeddings():
    return get_embedding_provider()


@pytest.fixture
def env(db_session, db_engine, monkeypatch, embeddings):
    ingest_policies(db_session, DEFAULT_POLICY_DIR, ChunkingConfig())
    materialize_embeddings(db_session, embeddings, batch_size=16)
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    tools = build_commerce_tools(
        ToolDependencies(session_scope=db_session_module.read_only_session)
    )

    @contextmanager
    def scope():
        yield db_session

    def retriever_for(day):
        noon = datetime(day.year, day.month, day.day, 12, tzinfo=UTC)
        return SemanticKnowledgeRetriever(scope, embeddings, clock=lambda: noon)

    return tools, retriever_for


def live(case_id, env):
    case = CASES[case_id]
    tools, retriever_for = env
    provider = get_llm_provider(provider="ollama", model=os.getenv("LIVE_RAG_MODEL") or None)
    return run_rag_case(case, provider, retriever_for, tools, tenant_id_for(case.tenant))


def test_live_current_bluepeak_delay_policy(env):
    o = live("bp-delay-current", env)
    assert o["ok"], (o["error_code"], o["error_detail"])
    assert o["retrieval_performed"], "model did not request policy retrieval"
    assert "policy://delayed-shipment-compensation/v2" in o["retrieved_sources"]
    assert "policy://delayed-shipment-compensation/v1" not in o["retrieved_sources"]
    assert o["citations"], "answer carried no current-run citation"
    assert o["tenant_isolation_ok"]


def test_live_historical_bluepeak_delay_policy(env):
    o = live("bp-delay-historical", env)
    assert o["ok"] or o["error_code"] in ("agent_grounding_error",), (
        o["error_code"],
        o["error_detail"],
    )
    assert o["retrieval_performed"], "model did not request policy retrieval"
    if "policy://delayed-shipment-compensation/v1" not in o["retrieved_sources"]:
        pytest.xfail(
            f"model did not extract the historical date (retrieval as_of={o['retrieval_as_of']})"
        )
    assert o["ok"] and o["citations"]
    assert all("/v2#" not in c for c in o["citations"] if "delayed-shipment" in c)


def test_live_pure_commerce_does_not_retrieve(env):
    o = live("ns-order-commerce-only", env)
    assert o["ok"], (o["error_code"], o["error_detail"])
    assert "get_order" in o["commerce_tools"] and o["retrieval_performed"] is False
    assert o["citations"] == []
