"""In-memory checkpointing: thread IDs, per-run resets, tenant-scoped isolation, and a real
checkpoint round trip of provider metadata. Ephemeral InMemorySaver only (no durable store).
"""

import json
import uuid
import warnings

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.assistant import AssistantError, AssistantLimits
from app.agent.assistant.executor import ToolExecutor
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant, checkpoint_thread_key
from app.agent.graph.state import (
    FAILURE_MARKER_TEXT,
    RUN_RESET,
    is_failure_marker,
    scope_digest,
)
from tests.assistant.fakes import (
    DownDatabase,
    ai_text,
    ai_tools,
    call,
    make_provider,
    offline_tools,
)

TENANT_A = uuid.uuid4()
TENANT_B = uuid.uuid4()
CTX_A = AgentContext(TENANT_A, "req-a")
CTX_B = AgentContext(TENANT_B, "req-b")


@pytest.fixture
def db():
    return DownDatabase()


@pytest.fixture
def saver():
    return InMemorySaver()


def build(db, saver, *script, limits=None):
    provider, model = make_provider(*script)
    assistant = CommerceGraphAssistant(
        provider, tools=offline_tools(db), limits=limits or AssistantLimits(), checkpointer=saver
    )
    return assistant, model


def state(assistant, ctx, thread):
    return assistant.graph.get_state(assistant.thread_config(ctx, thread)).values


def storage_blob(saver: InMemorySaver) -> bytes:
    return (
        repr(saver.storage).encode()
        + repr(dict(saver.writes)).encode()
        + repr(saver.blobs).encode()
    )


# --- thread id rules ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("with_saver", "thread_id", "detail"),
    [
        (True, None, "thread_id_required"),
        (False, "t1", "thread_id_without_checkpointer"),
        (True, "", "thread_id"),
        (True, "../other", "thread_id"),
        (True, "x" * 65, "thread_id"),
        (True, 42, "thread_id"),
    ],
)
def test_thread_id_rules(db, with_saver, thread_id, detail):
    provider, model = make_provider(ai_text("never"))
    assistant = CommerceGraphAssistant(
        provider,
        tools=offline_tools(db),
        limits=AssistantLimits(),
        checkpointer=InMemorySaver() if with_saver else None,
    )
    with pytest.raises(AssistantError) as exc:
        assistant.run("hi", CTX_A, thread_id=thread_id)
    assert (exc.value.code, exc.value.detail) == ("agent_input_invalid", detail)
    assert model.invocations == []


def test_checkpoint_key_is_tenant_scoped_and_opaque():
    ka = checkpoint_thread_key(TENANT_A, "shared")
    kb = checkpoint_thread_key(TENANT_B, "shared")
    assert ka != kb and ka.startswith("cg1-") and len(ka) == 68
    assert ka == checkpoint_thread_key(TENANT_A, "shared")  # deterministic
    assert checkpoint_thread_key(TENANT_A, "other") != ka
    for raw in (str(TENANT_A), TENANT_A.hex, "shared"):
        assert raw not in ka


# --- same thread: continuation and retrieval -----------------------------------------------
def test_same_thread_continues_with_the_system_prompt_once(db, saver):
    assistant, model = build(
        db,
        saver,
        ai_tools(call("get_order", "c1", order_number="ORD-1001")),
        ai_text("Order ORD-1001: data service unavailable."),
        ai_text("You asked about ORD-1001."),
    )
    first = assistant.run("Show me order ORD-1001", CTX_A, thread_id="t1")
    second = assistant.run("What did I ask?", CTX_A, thread_id="t1")

    sent = model.invocations[2].messages  # first model call of run 2
    assert [type(m) for m in sent] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        HumanMessage,
    ]
    assert sum(isinstance(m, SystemMessage) for m in sent) == 1
    assert (sent[1].content, sent[-1].content) == ("Show me order ORD-1001", "What did I ask?")
    assert (first.model_calls, len(first.tool_calls)) == (2, 1)
    assert (second.model_calls, second.tool_calls) == (1, [])  # per-run accounting

    values = state(assistant, CTX_A, "t1")  # retrievable for the same (tenant, thread)
    assert len(values["messages"]) == 7 and values["answer"] == "You asked about ORD-1001."
    assert sum(isinstance(m, SystemMessage) for m in values["messages"]) == 1


def test_per_run_fields_reset_while_messages_continue(db, saver):
    limits = AssistantLimits(max_model_rounds=2)
    assistant, _ = build(
        db,
        saver,
        ai_tools(call("list_delayed_shipments", "s1")),
        ai_tools(call("list_delayed_shipments", "s2")),  # final round -> limit error
        ai_text("fresh answer"),
        limits=limits,
    )
    with pytest.raises(AssistantError) as exc:
        assistant.run("loop", CTX_A, thread_id="t1")
    assert exc.value.detail == "max_model_rounds"
    before = state(assistant, CTX_A, "t1")
    assert before["error"]["detail"] == "max_model_rounds"
    assert before["model_calls"] == 2 and before["seen_tool_call_ids"] == ["s1"]
    assert len(before["tool_calls"]) == 1

    result = assistant.run("hello again", CTX_A, thread_id="t1")
    after = state(assistant, CTX_A, "t1")
    assert result.answer == "fresh answer" and result.model_calls == 1
    assert (
        after["error"],
        after["pending"],
        after["invalid_tool_calls"],
        after["tool_calls"],
        after["seen_tool_call_ids"],
        after["model_calls"],
    ) == (None, None, [], [], [], 1)
    assert len(after["messages"]) == len(before["messages"]) + 2  # history intentionally kept
    assert after["scope_digest"] == before["scope_digest"]


def test_invalid_call_summaries_reset_per_run(db, saver):
    assistant, _ = build(db, saver, ai_text("{}"), ai_text("real answer"))
    with pytest.raises(AssistantError):
        assistant.run("x", CTX_A, thread_id="t1")
    assert len(state(assistant, CTX_A, "t1")["invalid_tool_calls"]) == 1
    assistant.run("y", CTX_A, thread_id="t1")
    assert state(assistant, CTX_A, "t1")["invalid_tool_calls"] == []


def test_leftover_pending_batch_is_cleared_by_the_next_run(db, saver, monkeypatch):
    real = ToolExecutor.execute
    calls = {"n": 0}

    def crash_once(self, call_, context, round_no):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("process died mid-batch")
        return real(self, call_, context, round_no)

    monkeypatch.setattr(ToolExecutor, "execute", crash_once)
    assistant, model = build(
        db, saver, ai_tools(call("get_order", "p1", order_number="A")), ai_text("next run")
    )
    with pytest.raises(RuntimeError):
        assistant.run("first", CTX_A, thread_id="t1")
    stuck = state(assistant, CTX_A, "t1")
    assert stuck["pending"] is not None  # approved batch checkpointed before the crash
    assert [type(m) for m in stuck["messages"]] == [SystemMessage, HumanMessage]

    assistant.run("second", CTX_A, thread_id="t1")
    after = state(assistant, CTX_A, "t1")
    assert after["pending"] is None
    sent = model.invocations[1].messages
    # The unexecuted AIMessage never entered history; the crashed turn is closed with the
    # generic failure marker before the new user message (no consecutive HumanMessages).
    assert [type(m) for m in sent] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert is_failure_marker(sent[2]) and sent[2].content == FAILURE_MARKER_TEXT
    assert "process died" not in json.dumps([m.model_dump() for m in sent], default=str)
    assert not any(isinstance(m, ToolMessage) for m in after["messages"])


# --- isolation ---------------------------------------------------------------------------------
def test_different_threads_never_share_state(db, saver):
    assistant, model = build(db, saver, ai_text("A1"), ai_text("B1"), ai_text("A2"))
    assistant.run("secret-thread-a", CTX_A, thread_id="thread-a")
    assistant.run("hello from b", CTX_A, thread_id="thread-b")
    assistant.run("again a", CTX_A, thread_id="thread-a")
    b_inputs = [m.content for m in model.invocations[1].messages]
    assert "secret-thread-a" not in b_inputs and len(b_inputs) == 2
    assert [m.content for m in state(assistant, CTX_A, "thread-b")["messages"]][1:] == [
        "hello from b",
        "B1",
    ]
    assert len(state(assistant, CTX_A, "thread-a")["messages"]) == 5


def test_same_thread_name_under_two_tenants_is_two_threads(db, saver):
    assistant, model = build(db, saver, ai_text("for A"), ai_text("for B"))
    assistant.run("northstar-private-question", CTX_A, thread_id="shared")
    assistant.run("bluepeak question", CTX_B, thread_id="shared")
    b_sent = [m.content for m in model.invocations[1].messages]
    assert "northstar-private-question" not in b_sent and len(b_sent) == 2
    a_values, b_values = state(assistant, CTX_A, "shared"), state(assistant, CTX_B, "shared")
    assert a_values["answer"] == "for A" and b_values["answer"] == "for B"
    assert a_values["scope_digest"] == scope_digest(TENANT_A)
    assert b_values["scope_digest"] == scope_digest(TENANT_B)


def test_mismatched_runtime_context_is_refused_before_any_model_call(db, saver):
    assistant, model = build(db, saver, ai_text("A answer"), ai_text("must never be produced"))
    assistant.run("northstar question", CTX_A, thread_id="t1")
    a_config = assistant.thread_config(CTX_A, "t1")
    values = assistant.graph.invoke(  # bypass the runner: A's thread key, B's trusted context
        {**RUN_RESET, "messages": [HumanMessage("bluepeak intruder")]}, a_config, context=CTX_B
    )
    assert values["error"] == {
        "code": "agent_thread_conflict",
        "message": "This conversation thread belongs to a different account.",
        "detail": "scope_mismatch",
    }
    assert len(model.invocations) == 1 and values["model_calls"] == 0
    assert values["scope_digest"] == scope_digest(TENANT_A)  # binding not overwritten


def test_raw_tenant_thread_and_request_ids_are_never_checkpointed(db, saver):
    assistant, _ = build(
        db,
        saver,
        ai_tools(call("get_order", "c1", order_number="ORD-1001", tenant_id=str(TENANT_B))),
        ai_text("done"),
    )
    assistant.run("Show me order ORD-1001", CTX_A, thread_id="raw-thread-name")
    blob = storage_blob(saver)
    assert str(TENANT_A).encode() not in blob and TENANT_A.hex.encode() not in blob
    assert b"raw-thread-name" not in blob and b"req-a" not in blob
    assert checkpoint_thread_key(TENANT_A, "raw-thread-name").encode() in blob
    # The smuggled tenant argument never became a summary argument either.
    [summary] = state(assistant, CTX_A, "raw-thread-name")["tool_calls"]
    assert summary["arguments"] == {"order_number": "ORD-1001"}
    assert summary["rejected_argument_names"] == ["tenant_id"]


def test_checkpointed_state_contains_only_registered_types(db, saver):
    assistant, _ = build(
        db,
        saver,
        ai_tools(call("get_order", "c1", order_number="A")),
        ai_text("done"),
        ai_text("again"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assistant.run("x", CTX_A, thread_id="t1")
        assistant.run("y", CTX_A, thread_id="t1")
        list(assistant.graph.get_state_history(assistant.thread_config(CTX_A, "t1")))
    assert not [w for w in caught if "unregistered type" in str(w.message)]


# --- provider metadata through a real checkpoint round trip -------------------------------
def test_provider_metadata_survives_model_checkpoint_tools_next_model(db, saver):
    sentinel = AIMessage(
        content=[
            {"type": "text", "text": ""},
            {"type": "function_call", "name": "get_invoice", "extras": {"signature": "SIG=="}},
        ],
        tool_calls=[call("get_invoice", "sig-1", invoice_number="INV-1001")],
        additional_kwargs={"__gemini_function_call_thought_signatures__": {"sig-1": "opaque=="}},
        response_metadata={"model_name": "gemini-3.8-flash", "finish_reason": "STOP"},
        usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        id="provider-run-1",
    )
    expected = sentinel.model_copy(deep=True)
    assistant, model = build(db, saver, sentinel, ai_text("done"), ai_text("follow-up"))
    assistant.run("invoice INV-1001", CTX_A, thread_id="t1")

    # The approved message crossed a checkpoint in `pending` (model node -> tools node).
    history = list(assistant.graph.get_state_history(assistant.thread_config(CTX_A, "t1")))
    [pending_snapshot] = [h for h in history if h.values.get("pending") is not None]
    assert pending_snapshot.next == ("tools",)
    assert pending_snapshot.values["pending"] == expected  # deserialized from the checkpoint

    # A NEW graph instance on the same saver only sees the checkpointed copy.
    provider2, model2 = make_provider(ai_text("follow-up"))
    fresh = CommerceGraphAssistant(
        provider2, tools=offline_tools(db), limits=AssistantLimits(), checkpointer=saver
    )
    fresh.run("and the amount?", CTX_A, thread_id="t1")
    passed = model2.invocations[0].messages[2]
    assert passed is not sentinel
    assert passed == expected
    assert passed.additional_kwargs == expected.additional_kwargs
    assert passed.response_metadata == expected.response_metadata
    assert passed.usage_metadata == expected.usage_metadata
    assert passed.content == expected.content and passed.id == "provider-run-1"
    # And within the original run the next model call got the same semantics.
    assert model.invocations[1].messages[2] == expected
