"""Current-turn grounding on checkpointed threads: history stays, citations do not carry over."""

import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.assistant import AssistantError, AssistantLimits
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.state import RUN_RESET, is_failure_marker
from tests.assistant.fakes import (
    DownDatabase,
    ai_text,
    ai_tools,
    call,
    make_provider,
    offline_tools,
)
from tests.rag.fakes import FakeRetriever, chunk, policy_messages, result, search

CTX = AgentContext(uuid.uuid4(), "req-cp")
V2 = "policy://delayed-shipment-compensation/v2#chunk-2"
V1 = "policy://refund-policy/v1#chunk-3"


def assistant_for(*script, retriever=None):
    provider, model = make_provider(*script)
    retriever = retriever or FakeRetriever()
    a = CommerceGraphAssistant(
        provider,
        tools=offline_tools(DownDatabase()),
        limits=AssistantLimits(),
        retriever=retriever,
        checkpointer=InMemorySaver(),
    )
    return a, model, retriever


def cite(c):
    return ai_text(f"Store credit is 15% [{c}].")


def test_run_reset_covers_the_new_per_run_fields():
    assert RUN_RESET["retrievals"] == [] and RUN_RESET["policy_sources"] == []
    assert RUN_RESET["policy_retrieval_status"] == "none" and RUN_RESET["citations"] == []
    assert RUN_RESET["pending_kind"] is None


def test_previous_turn_citation_is_stale_without_fresh_retrieval():
    # run 2 answers from memory with the old citation, and again after the one correction
    a, model, retriever = assistant_for(ai_tools(search()), cite(V2), cite(V2), cite(V2))
    first = a.run("What compensation applies?", CTX, thread_id="t")
    assert [c.citation for c in first.citations] == [V2]
    with pytest.raises(AssistantError) as exc:
        a.run("And for a second parcel?", CTX, thread_id="t")
    assert (exc.value.code, exc.value.detail) == ("agent_grounding_error", "stale_citation")
    assert len(retriever.calls) == 1
    # old policy context was still in the model's history - it just does not count
    assert len(policy_messages(model.invocations[2].messages)) == 1
    state = a.graph.get_state(a.thread_config(CTX, "t")).values
    assert is_failure_marker(state["messages"][-1])
    assert state["policy_sources"] == [] and state["citations"] == []


def test_fresh_retrieval_on_the_next_turn_is_grounded():
    retriever = FakeRetriever([result(chunk()), result(chunk())])
    a, _, _ = assistant_for(
        ai_tools(search()), cite(V2), ai_tools(search("again")), cite(V2), retriever=retriever
    )
    a.run("What compensation applies?", CTX, thread_id="t")
    second = a.run("Remind me?", CTX, thread_id="t")
    assert [c.citation for c in second.citations] == [V2]
    assert [r.round for r in second.retrievals] == [1]  # only this run's retrieval
    assert len(retriever.calls) == 2


def test_catalog_resets_on_a_commerce_only_follow_up():
    a, _, _ = assistant_for(
        ai_tools(search()),
        cite(V2),
        ai_tools(call("get_order", order_number="ORD-1001")),
        ai_text("Unavailable."),
    )
    a.run("policy?", CTX, thread_id="t")
    second = a.run("order?", CTX, thread_id="t")
    assert second.retrievals == [] and second.citations == []
    state = a.graph.get_state(a.thread_config(CTX, "t")).values
    assert state["policy_sources"] == [] and state["policy_retrieval_status"] == "none"


def test_commerce_call_after_retrieval_rule_is_per_run():
    a, _, _ = assistant_for(
        ai_tools(search()),
        cite(V2),
        ai_tools(call("get_order", order_number="ORD-1001")),
        ai_text("ok"),
    )
    a.run("policy?", CTX, thread_id="t")
    assert a.run("order?", CTX, thread_id="t").tool_calls[0].tool == "get_order"


def test_retrieval_artifact_survives_the_checkpoint_round_trip():
    a, _, _ = assistant_for(ai_tools(search()), cite(V2))
    a.run("policy?", CTX, thread_id="t")
    state = a.graph.get_state(a.thread_config(CTX, "t")).values
    [tm] = policy_messages(state["messages"])
    assert tm.artifact == {"policy_citations": [V2]}


def test_stale_detection_uses_only_this_threads_history():
    """A citation retrieved in ANOTHER thread is simply not retrieved (not stale)."""
    a, _, _ = assistant_for(ai_tools(search()), cite(V2), cite(V2), cite(V2))
    a.run("policy?", CTX, thread_id="t1")
    with pytest.raises(AssistantError) as exc:
        a.run("policy?", CTX, thread_id="t2")
    assert exc.value.detail == "citation_not_retrieved"


def test_other_version_from_an_earlier_turn_is_rejected():
    retriever = FakeRetriever(
        [result(chunk(key="refund-policy", version=1, index=3)), result(chunk())]
    )
    a, _, _ = assistant_for(
        ai_tools(search()),
        cite(V1),
        ai_tools(search()),
        ai_text(f"Now [{V2}] and earlier [{V1}]."),
        retriever=retriever,
    )
    a.run("q", CTX, thread_id="t")
    with pytest.raises(AssistantError) as exc:
        a.run("q2", CTX, thread_id="t")
    assert exc.value.detail == "stale_citation"
