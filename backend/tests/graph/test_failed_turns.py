"""Failed turns on checkpointed threads are closed with a generic synthetic assistant marker.

Policy under test: when a checkpointed run's user message has entered thread state and the
run ends in an application/graph error, the runner appends FAILURE_MARKER_TEXT as an
AIMessage (no model call, no counters), keeps earlier checkpoints, keeps the real error in
graph state, and still raises the original AssistantError. One-shot runs are unchanged.
"""

import json
import logging
import uuid

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

import app.agent.graph.runner as runner_module
from app.agent.assistant import AssistantError, AssistantLimits
from app.agent.assistant.executor import ToolExecutionError, ToolExecutor
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.state import FAILURE_MARKER_TEXT, is_failure_marker
from app.agent.prompts import assistant as assistant_prompt
from tests.assistant.fakes import (
    DownDatabase,
    ai_text,
    ai_tools,
    call,
    make_provider,
    offline_tools,
)

TENANT_A, TENANT_B = uuid.uuid4(), uuid.uuid4()
CTX_A = AgentContext(TENANT_A, "req-a")
CTX_B = AgentContext(TENANT_B, "req-b")
REQ = httpx.Request("POST", "http://localhost:11434/api/chat")
D = AssistantLimits()


@pytest.fixture
def db():
    return DownDatabase()


@pytest.fixture
def saver():
    return InMemorySaver()


def build(db, saver, *script, limits=None):
    provider, model = make_provider(*script)
    assistant = CommerceGraphAssistant(
        provider, tools=offline_tools(db), limits=limits or D, checkpointer=saver
    )
    return assistant, model


def values(assistant, ctx, thread="t1"):
    return assistant.graph.get_state(assistant.thread_config(ctx, thread)).values


def kinds(messages: list[BaseMessage]) -> list[str]:
    return ["MARKER" if is_failure_marker(m) else type(m).__name__ for m in messages]


def assert_coherent(messages: list[BaseMessage]) -> None:
    """No consecutive user messages; tool results are never followed by a user message."""
    for prev, nxt in zip(messages, messages[1:], strict=False):
        if isinstance(nxt, HumanMessage):
            assert not isinstance(prev, HumanMessage | ToolMessage), kinds(messages)


def _tool_crash(self, call_, context, round_no):
    raise ToolExecutionError("lost id")


FAILURES = {
    # name: (script, limits, expected code, expected detail, expected history after the turn)
    "provider_timeout": (
        lambda: [httpx.ReadTimeout("t", request=REQ), httpx.ReadTimeout("t", request=REQ)],
        D,
        "llm_timeout",
        "llm",
        ["HumanMessage", "MARKER"],
    ),
    "provider_internal_error": (
        lambda: [ValueError("psycopg DSN postgresql://u:pw@db AIzaSECRET Traceback internals")],
        D,
        "llm_internal_error",
        "llm",
        ["HumanMessage", "MARKER"],
    ),
    "protocol_error_bare_braces": (
        lambda: [ai_text("{}")],
        D,
        "agent_protocol_error",
        "protocol_artifact",
        ["HumanMessage", "MARKER"],
    ),
    "protocol_error_invalid_tool_calls": (
        lambda: [
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "type": "invalid_tool_call",
                        "id": "x",
                        "name": "get_order",
                        "args": "{",
                        "error": "bad json",
                    }
                ],
            )
        ],
        D,
        "agent_protocol_error",
        "invalid_tool_calls",
        ["HumanMessage", "MARKER"],
    ),
    "empty_answer": (
        lambda: [ai_text("")],
        D,
        "agent_empty_answer",
        None,
        ["HumanMessage", "MARKER"],
    ),
    "limit_per_turn": (
        lambda: [ai_tools(*[call("get_order", f"p{i}", order_number="A") for i in range(5)])],
        AssistantLimits(max_tool_calls_per_turn=4),
        "agent_limit_exceeded",
        "max_tool_calls_per_turn",
        ["HumanMessage", "MARKER"],
    ),
    "limit_final_round_after_a_tool_round": (
        lambda: [
            ai_tools(call("list_delayed_shipments", "r1")),
            ai_tools(call("list_delayed_shipments", "r2")),
        ],
        AssistantLimits(max_model_rounds=2),
        "agent_limit_exceeded",
        "max_model_rounds",
        ["HumanMessage", "AIMessage", "ToolMessage", "MARKER"],
    ),
}


@pytest.mark.parametrize("name", list(FAILURES))
def test_failed_checkpointed_turn_is_closed_and_original_error_raised(db, saver, name):
    script, limits, code, detail, tail = FAILURES[name]
    assistant, model = build(db, saver, *script(), limits=limits)
    with pytest.raises(AssistantError) as exc:
        assistant.run("first question", CTX_A, thread_id="t1")
    assert (exc.value.code, exc.value.detail) == (code, detail)

    v = values(assistant, CTX_A)
    assert kinds(v["messages"]) == ["SystemMessage", *tail]
    marker = v["messages"][-1]
    assert marker.content == FAILURE_MARKER_TEXT and not marker.tool_calls
    # Real outcome preserved for the application; the marker is not a model call.
    assert (v["error"]["code"], v["error"]["detail"]) == (code, detail)
    assert v["model_calls"] == exc.value.model_calls == _model_attempt_calls(model, code)
    assert (
        v["pending"] is None
        and assistant.graph.get_state(assistant.thread_config(CTX_A, "t1")).next == ()
    )


def _model_attempt_calls(model, code) -> int:
    # llm_timeout is retried once by the provider: 2 invocations, 1 logical model call.
    return 1 if code == "llm_timeout" else len(model.invocations)


@pytest.mark.parametrize("name", list(FAILURES))
def test_raised_error_matches_the_one_shot_error(db, name):
    """Checkpointing changes thread history only, never the error the caller sees."""
    script, limits, *_ = FAILURES[name]
    errors = []
    for saver in (None, InMemorySaver()):
        provider, _ = make_provider(*script())
        a = CommerceGraphAssistant(
            provider, tools=offline_tools(db), limits=limits, checkpointer=saver
        )
        with pytest.raises(AssistantError) as exc:
            a.run("q", CTX_A, thread_id="t1" if saver else None)
        e = exc.value
        errors.append(
            (
                e.code,
                e.detail,
                e.message,
                e.model_calls,
                [c.model_dump(exclude={"duration_ms"}) for c in e.tool_calls],
                [c.model_dump() for c in e.invalid_tool_calls],
            )
        )
    assert errors[0] == errors[1]


@pytest.mark.parametrize("name", list(FAILURES))
def test_continued_thread_after_failure_is_coherent(db, saver, name):
    script, limits, *_ = FAILURES[name]
    assistant, model = build(db, saver, *script(), ai_text("second answer"), limits=limits)
    with pytest.raises(AssistantError):
        assistant.run("first question", CTX_A, thread_id="t1")
    before = len(model.invocations)
    result = assistant.run("second question", CTX_A, thread_id="t1")

    sent = model.invocations[before].messages
    assert_coherent(sent)
    assert sum(isinstance(m, SystemMessage) for m in sent) == 1
    marker_at = next(i for i, m in enumerate(sent) if is_failure_marker(m))
    assert isinstance(sent[marker_at + 1], HumanMessage)
    assert sent[marker_at + 1].content == "second question"
    assert result.answer == "second answer" and result.model_calls == 1
    assert_coherent(values(assistant, CTX_A)["messages"])


def test_tool_result_protocol_failure_closes_the_turn(db, saver, monkeypatch):
    monkeypatch.setattr(ToolExecutor, "execute", _tool_crash)
    assistant, _ = build(db, saver, ai_tools(call("get_order", "a", order_number="A")))
    with pytest.raises(AssistantError) as exc:
        assistant.run("first", CTX_A, thread_id="t1")
    assert (exc.value.code, exc.value.detail) == ("agent_protocol_error", "tool_result")
    v = values(assistant, CTX_A)
    # The unexecuted tool-calling AIMessage never entered history.
    assert kinds(v["messages"]) == ["SystemMessage", "HumanMessage", "MARKER"]
    assert v["error"]["detail"] == "tool_result" and v["model_calls"] == 1


def test_recursion_limit_failure_closes_the_turn_and_records_the_error(db, saver, monkeypatch):
    monkeypatch.setattr(runner_module, "recursion_limit_for", lambda _limits: 2)
    assistant, _ = build(db, saver, *[ai_tools(call("list_delayed_shipments")) for _ in range(5)])
    with pytest.raises(AssistantError) as exc:
        assistant.run("loop", CTX_A, thread_id="t1")
    assert (exc.value.code, exc.value.detail) == ("agent_limit_exceeded", "recursion_limit")
    v = values(assistant, CTX_A)
    assert kinds(v["messages"])[-1] == "MARKER"
    assert v["error"] == {
        "code": "agent_limit_exceeded",
        "message": None,
        "detail": "recursion_limit",
    }
    assert v["pending"] is None
    assert assistant.graph.get_state(assistant.thread_config(CTX_A, "t1")).next == ()


def test_marker_contains_no_internal_error_text(db, saver):
    secret = "psycopg DSN postgresql://u:pw@db AIzaSECRET Traceback internals"
    assistant, _ = build(db, saver, ValueError(secret))
    with pytest.raises(AssistantError):
        assistant.run("q", CTX_A, thread_id="t1")
    marker = values(assistant, CTX_A)["messages"][-1]
    assert is_failure_marker(marker)
    assert marker.content == FAILURE_MARKER_TEXT == "The previous request could not be completed."
    assert (marker.tool_calls, marker.additional_kwargs, marker.usage_metadata) == ([], {}, None)
    blob = json.dumps(marker.model_dump(), default=str)
    for leak in (
        "psycopg",
        "postgresql",
        "AIza",
        "Traceback",
        "internals",
        "llm_",
        "agent_",
        "ValueError",
        str(TENANT_A),
        TENANT_A.hex,
        "req-a",
        "t1",
        assistant_prompt.SYSTEM_PROMPT[:40],
    ):
        assert leak not in blob


def test_marker_does_not_increment_model_calls(db, saver):
    assistant, model = build(
        db, saver, ai_tools(call("get_order", "a", order_number="A")), ai_text("{}")
    )
    with pytest.raises(AssistantError) as exc:
        assistant.run("q", CTX_A, thread_id="t1")
    history = list(assistant.graph.get_state_history(assistant.thread_config(CTX_A, "t1")))
    latest, before_marker = history[0], history[1]
    assert is_failure_marker(latest.values["messages"][-1])
    assert not any(is_failure_marker(m) for m in before_marker.values["messages"])
    assert latest.values["model_calls"] == before_marker.values["model_calls"] == 2
    assert exc.value.model_calls == 2 == len(model.invocations)
    assert latest.metadata["source"] == "update"  # written by the runner, not a graph step


def test_failed_run_checkpoints_are_kept(db, saver):
    assistant, _ = build(
        db, saver, ai_tools(call("get_order", "a", order_number="A")), ai_text("{}")
    )
    with pytest.raises(AssistantError):
        assistant.run("q", CTX_A, thread_id="t1")
    history = list(assistant.graph.get_state_history(assistant.thread_config(CTX_A, "t1")))
    assert len(history) >= 4  # input, model, tools, model, then the marker update
    # The successful tool round of the failed run is still in the kept checkpoints and in
    # the final history: nothing was rolled back or deleted.
    assert any(h.values.get("pending") is not None for h in history)
    assert kinds(history[0].values["messages"]) == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "MARKER",
    ]


def test_invalid_input_on_a_thread_writes_nothing(db, saver):
    assistant, model = build(db, saver, ai_text("ok"))
    assistant.run("hello", CTX_A, thread_id="t1")
    before = values(assistant, CTX_A)["messages"]
    for bad in ("   ", "x" * 4001):
        with pytest.raises(AssistantError) as exc:
            assistant.run(bad, CTX_A, thread_id="t1")
        assert exc.value.code == "agent_input_invalid"
    assert values(assistant, CTX_A)["messages"] == before and len(model.invocations) == 1


def test_successful_turns_get_no_marker(db, saver):
    assistant, _ = build(
        db,
        saver,
        ai_tools(call("get_order", "a", order_number="A")),
        ai_text("done"),
        ai_text("again"),
    )
    assistant.run("one", CTX_A, thread_id="t1")
    assistant.run("two", CTX_A, thread_id="t1")
    assert not any(is_failure_marker(m) for m in values(assistant, CTX_A)["messages"])


def test_one_shot_failure_writes_no_conversation_state(db, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("one-shot runs must not touch checkpoint state")

    monkeypatch.setattr(CompiledStateGraph, "update_state", forbidden)
    monkeypatch.setattr(CompiledStateGraph, "get_state", forbidden)
    provider, _ = make_provider(ai_text("{}"))
    a = CommerceGraphAssistant(provider, tools=offline_tools(db), limits=D)
    with pytest.raises(AssistantError) as exc:
        a.run("q", CTX_A)
    assert exc.value.code == "agent_protocol_error"


def test_failure_isolation_across_threads_and_tenants(db, saver):
    assistant, model = build(db, saver, ai_text("{}"), ai_text("B ok"), ai_text("A2 ok"))
    with pytest.raises(AssistantError):
        assistant.run("northstar failing question", CTX_A, thread_id="shared")
    assistant.run("bluepeak question", CTX_B, thread_id="shared")
    assistant.run("northstar other thread", CTX_A, thread_id="other")

    for ctx, thread in ((CTX_B, "shared"), (CTX_A, "other")):
        msgs = values(assistant, ctx, thread)["messages"]
        assert not any(is_failure_marker(m) for m in msgs)
        assert "northstar failing question" not in [m.content for m in msgs]
    for inv in model.invocations[1:]:
        assert not any(is_failure_marker(m) for m in inv.messages)
    assert kinds(values(assistant, CTX_A, "shared")["messages"]) == [
        "SystemMessage",
        "HumanMessage",
        "MARKER",
    ]


def test_no_marker_is_written_into_a_thread_bound_to_another_tenant(db, saver, monkeypatch):
    """Simulated key collision: B's failing run must not close (write into) A's thread."""
    assistant, _ = build(db, saver, ai_text("A answer"))
    assistant.run("northstar question", CTX_A, thread_id="t1")
    a_config = assistant.thread_config(CTX_A, "t1")
    monkeypatch.setattr(CommerceGraphAssistant, "thread_config", lambda self, c, t: a_config)
    with pytest.raises(AssistantError) as exc:
        assistant.run("bluepeak intruder", CTX_B, thread_id="t1")
    assert exc.value.code == "agent_thread_conflict"
    msgs = assistant.graph.get_state(a_config).values["messages"]
    assert not any(is_failure_marker(m) for m in msgs)


def test_failed_run_log_reports_turn_closure_safely(db, saver, caplog):
    assistant, _ = build(db, saver, ai_text("{}"))
    with caplog.at_level(logging.INFO), pytest.raises(AssistantError):
        assistant.run("PRIVATE-USER-TEXT", CTX_A, thread_id="t1")
    [rec] = [r for r in caplog.records if r.name == "app.agent.graph"]
    assert (rec.outcome, rec.failed_turn_closed, rec.checkpointed) == (
        "agent_protocol_error",
        True,
        True,
    )
    assert "PRIVATE-USER-TEXT" not in caplog.text and FAILURE_MARKER_TEXT not in caplog.text
