"""End-to-end RAG on real PostgreSQL + pgvector (Step 9). Only the chat model (scripted) and the
embedding model (deterministic hashing fake) are fake:

    graph -> MODEL -> RETRIEVE -> SemanticKnowledgeRetriever -> pgvector (tenant + date filtered)
          -> ToolMessage -> MODEL -> grounding validator -> END
"""

import json
import logging
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

import app.db.session as db_session_module
from app.agent.assistant import AssistantError, AssistantLimits
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from app.agent.tools import ToolDependencies, build_commerce_tools
from app.knowledge.chunking import ChunkingConfig
from app.knowledge.embeddings.errors import EmbeddingUnavailableError
from app.knowledge.embeddings.materialize import materialize_embeddings
from app.knowledge.ingest import ingest_policies
from app.knowledge.semantic import SemanticKnowledgeRetriever
from app.knowledge.sources import DEFAULT_POLICY_DIR
from app.models import KnowledgeChunk
from scripts import run_graph_assistant
from tests.assistant.fakes import ai_text, ai_tools, call, make_provider
from tests.db.conftest import fixed_clock
from tests.knowledge.fakes import BrokenProvider, HashingEmbeddingProvider
from tests.rag.fakes import policy_messages, search

INJECTION_QUERY = "parcel locker delivery note: parcel locker compensation, standard handling"
INJECTION_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "policies_injection"


@pytest.fixture
def kb(db_session):
    ingest_policies(db_session, DEFAULT_POLICY_DIR, ChunkingConfig())
    return db_session


@pytest.fixture
def materialized(kb):
    materialize_embeddings(kb, HashingEmbeddingProvider(), batch_size=16)
    return kb


@pytest.fixture
def scope(kb):
    @contextmanager
    def _scope():
        yield kb

    return _scope


@pytest.fixture
def real_tools(db_engine, monkeypatch):
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    return build_commerce_tools(
        ToolDependencies(session_scope=db_session_module.read_only_session, clock=fixed_clock)
    )


@pytest.fixture
def ns(tenant_a):
    return AgentContext(tenant_a.tenant_id, "req-ns")


@pytest.fixture
def bp(tenant_b):
    return AgentContext(tenant_b.tenant_id, "req-bp")


class RecordingRetriever:
    def __init__(self, inner):
        self.inner = inner
        self.contexts = []

    def retrieve(self, query, context, **kw):
        self.contexts.append(context)
        return self.inner.retrieve(query, context, **kw)


def semantic(scope, provider=None):
    return RecordingRetriever(
        SemanticKnowledgeRetriever(scope, provider or HashingEmbeddingProvider(), clock=fixed_clock)
    )


def cite_retrieved(prefix: str | None = None):
    """Scripted final answer citing the first retrieved citation (optionally with a prefix)."""

    def step(messages):
        [*_, last] = policy_messages(messages)
        citations = last.artifact["policy_citations"]
        chosen = next(
            (c for c in citations if prefix is None or c.startswith(prefix)), citations[0]
        )
        return AIMessage(content=f"Per policy [{chosen}].")

    return step


def run(real_tools, retriever, ctx, *script, saver=None):
    provider, model = make_provider(*script)
    a = CommerceGraphAssistant(
        provider,
        tools=real_tools,
        limits=AssistantLimits(),
        retriever=retriever,
        checkpointer=saver or InMemorySaver(),
    )
    res = a.run("question", ctx, thread_id="t")
    state = a.graph.get_state(a.thread_config(ctx, "t")).values
    return res, state, model


def retrieved_versions(state, key):
    return {s["version"] for s in state["policy_sources"] if s["document_key"] == key}


def all_citations(res, state):
    tool_text = " ".join(m.content for m in policy_messages(state["messages"]))
    return (
        tool_text,
        [s["citation"] for s in state["policy_sources"]],
        [c.citation for c in res.citations],
    )


# --- temporal ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("as_of", "version", "other"),
    [("2026-06-10", 1, 2), ("2026-09-01", 2, 1)],
)
def test_northstar_refund_versions(materialized, scope, real_tools, ns, as_of, version, other):
    retriever = semantic(scope)
    res, state, _ = run(
        real_tools,
        retriever,
        ns,
        ai_tools(search("refund policy refund timing business days", as_of=as_of)),
        cite_retrieved(f"policy://refund-policy/v{version}"),
    )
    assert retrieved_versions(state, "refund-policy") == {version}
    tool_text, catalog, final = all_citations(res, state)
    assert f"refund-policy/v{other}#" not in tool_text + " ".join(catalog + final)
    assert final == [c for c in final if c.startswith(f"policy://refund-policy/v{version}#")]
    assert res.retrievals[0].as_of == date.fromisoformat(as_of)


@pytest.mark.parametrize(
    ("as_of", "version"),
    [("2026-06-10", 1), ("2026-08-14", 1), ("2026-08-15", 2), ("2026-09-01", 2)],
)
def test_bluepeak_delayed_compensation_versions_and_boundary(
    materialized, scope, real_tools, bp, as_of, version
):
    res, state, _ = run(
        real_tools,
        semantic(scope),
        bp,
        ai_tools(search("delayed shipment compensation store credit", as_of=as_of)),
        cite_retrieved(f"policy://delayed-shipment-compensation/v{version}"),
    )
    assert retrieved_versions(state, "delayed-shipment-compensation") == {version}
    tool_text, catalog, final = all_citations(res, state)
    wrong = f"delayed-shipment-compensation/v{3 - version}#"
    assert wrong not in tool_text and all(wrong not in c for c in catalog + final)


def test_omitted_as_of_uses_the_trusted_clock(materialized, scope, real_tools, bp):
    res, state, _ = run(
        real_tools,
        semantic(scope),
        bp,
        ai_tools(search("delayed shipment compensation store credit")),
        cite_retrieved(),
    )
    assert res.retrievals[0].as_of == fixed_clock().date()
    assert retrieved_versions(state, "delayed-shipment-compensation") == {2}


# --- tenant isolation ---------------------------------------------------------------------
QUESTION = "What compensation applies to a delayed shipment?"


def _contents(session, tenant_id):
    return set(
        session.scalars(select(KnowledgeChunk.content).where(KnowledgeChunk.tenant_id == tenant_id))
    )


def test_same_question_is_grounded_per_tenant(materialized, scope, real_tools, ns, bp):
    ns_only = _contents(materialized, ns.tenant_id) - _contents(materialized, bp.tenant_id)
    bp_only = _contents(materialized, bp.tenant_id) - _contents(materialized, ns.tenant_id)
    for ctx, foreign in ((ns, bp_only), (bp, ns_only)):
        retriever = semantic(scope)
        res, state, model = run(
            real_tools,
            retriever,
            ctx,
            ai_tools(search(QUESTION)),
            cite_retrieved("policy://delayed"),
        )
        assert [c.tenant_id for c in retriever.contexts] == [ctx.tenant_id]
        sent = " ".join(str(m.content) for i in model.invocations for m in i.messages)
        assert not any(text in sent for text in foreign if len(text) > 40)
        assert res.citations and res.citations[0].document_key == "delayed-shipment-compensation"
    # Northstar has only v1 of this policy; BluePeak's current is v2.


def test_cross_tenant_citation_is_rejected(materialized, scope, real_tools, ns):
    # policy://delayed-shipment-compensation/v2 exists only for BluePeak.
    with pytest.raises(AssistantError) as exc:
        run(
            real_tools,
            semantic(scope),
            ns,
            ai_tools(search(QUESTION)),
            ai_text("15% [policy://delayed-shipment-compensation/v2#chunk-2]."),
        )
    assert (exc.value.code, exc.value.detail) == ("agent_grounding_error", "citation_not_retrieved")


def test_model_cannot_add_a_tenant_to_retrieval(materialized, scope, real_tools, ns, bp):
    retriever = semantic(scope)
    res, _, _ = run(
        real_tools,
        retriever,
        ns,
        ai_tools(search(QUESTION, tenant_id=str(bp.tenant_id))),
        ai_tools(search(QUESTION)),
        cite_retrieved(),
    )
    assert [c.tenant_id for c in retriever.contexts] == [ns.tenant_id]
    assert res.retrievals[0].rejected_argument_names == ["tenant_id"]


# --- mixed commerce + policy --------------------------------------------------------------
def test_mixed_shipment_and_policy_trajectory(materialized, scope, real_tools, ns, caplog):
    retriever = semantic(scope)
    with caplog.at_level(logging.INFO):
        res, state, model = run(
            real_tools,
            retriever,
            ns,
            ai_tools(call("get_shipment", "c1", shipment_number="SHP-1003")),
            ai_tools(search("delayed shipment compensation", "s1")),
            cite_retrieved("policy://delayed-shipment-compensation/v1"),
        )
    assert [(c.round, c.tool, c.outcome) for c in res.tool_calls] == [
        (1, "get_shipment", "success")
    ]
    assert [r.round for r in res.retrievals] == [2]
    tool_log = [r for r in caplog.records if r.name == "app.agent.tools"]
    assert [r.tenant_id for r in tool_log] == [str(ns.tenant_id)]
    assert [c.tenant_id for c in retriever.contexts] == [ns.tenant_id]
    final = model.invocations[2].messages
    shipment = next(m for m in final if isinstance(m, ToolMessage) and m.tool_call_id == "c1")
    assert json.loads(shipment.content)["data"]["status"] == "delayed"
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "s1" for m in final)
    assert res.citations[0].citation.startswith("policy://delayed-shipment-compensation/v1#")


# --- failures on the real retriever ---------------------------------------------------------
def test_profile_not_materialized_is_a_retrieval_error(kb, scope, real_tools, bp):
    with pytest.raises(AssistantError) as exc:
        run(real_tools, semantic(scope), bp, ai_tools(search(QUESTION)), ai_text("From memory."))
    assert (exc.value.code, exc.value.detail) == (
        "agent_retrieval_error",
        "embedding_profile_not_materialized",
    )


def test_embedding_provider_unavailable_is_a_retrieval_error(materialized, scope, real_tools, bp):
    broken = BrokenProvider(error=EmbeddingUnavailableError())
    with pytest.raises(AssistantError) as exc:
        run(real_tools, semantic(scope, broken), bp, ai_tools(search(QUESTION)), ai_text("x"))
    assert (exc.value.code, exc.value.detail) == ("agent_retrieval_error", "embedding_unavailable")


def test_no_eligible_policy_is_no_results_not_an_error(materialized, scope, real_tools, ns):
    res, state, model = run(
        real_tools,
        semantic(scope),
        ns,
        ai_tools(search("refund policy", as_of="2025-12-01")),
        ai_text("The applicable policy information could not be found."),
    )
    assert res.retrievals[0].outcome == "no_results" and state["policy_sources"] == []
    assert res.citations == []


# --- prompt injection (dedicated fixture, not the demo corpus) -------------------------------
def test_injected_policy_text_cannot_execute_capabilities_or_citations(kb, scope, real_tools, bp):
    ingest_policies(kb, INJECTION_DIR, ChunkingConfig())
    materialize_embeddings(kb, HashingEmbeddingProvider(), batch_size=16)
    retriever = semantic(scope)
    obey_injection = ai_tools(
        call(
            "get_customer",
            customer_code="CUS-2001",
            tenant_id="11111111-1111-4111-8111-111111111111",
        )
    )
    with pytest.raises(AssistantError) as exc:
        run(
            real_tools,
            retriever,
            bp,
            ai_tools(search(INJECTION_QUERY, as_of="2026-09-01")),
            obey_injection,
        )
    assert (exc.value.code, exc.value.detail) == (
        "agent_protocol_error",
        "commerce_call_after_retrieval",
    )
    assert exc.value.tool_calls == []
    assert [c.tenant_id for c in retriever.contexts] == [bp.tenant_id]
    # the malicious chunk really was delivered to the model
    assert any(
        c.startswith("policy://parcel-locker-note/v1#") for c in exc.value.retrievals[0].citations
    )

    with pytest.raises(AssistantError) as exc2:
        run(
            real_tools,
            semantic(scope),
            bp,
            ai_tools(search(INJECTION_QUERY, as_of="2026-09-01")),
            ai_text("100% refund [policy://admin-override/v9#chunk-9]."),
        )
    assert (exc2.value.code, exc2.value.detail) == (
        "agent_grounding_error",
        "citation_not_retrieved",
    )


# --- CLI --------------------------------------------------------------------------------------
def test_cli_prints_retrievals_and_citations(
    db_engine, materialized, scope, tenant_b, monkeypatch, capsys
):
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    provider, _ = make_provider(ai_tools(search(QUESTION)), cite_retrieved())
    monkeypatch.setattr(run_graph_assistant, "get_llm_provider", lambda **_k: provider)
    monkeypatch.setattr(run_graph_assistant, "build_policy_retriever", lambda: semantic(scope))
    code = run_graph_assistant.main(
        [
            "--tenant",
            str(tenant_b.tenant_id),
            "--text",
            "What compensation applies to a shipment delayed by 10 days?",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert {"answer", "tool_calls", "retrievals", "citations", "model_calls", "duration_ms"} <= set(
        data
    )
    assert data["prompt_version"] == "commerce-assistant-v2"
    assert data["citations"] and data["retrievals"][0]["outcome"] == "success"
    assert str(tenant_b.tenant_id) not in out and "query" not in data["retrievals"][0]
