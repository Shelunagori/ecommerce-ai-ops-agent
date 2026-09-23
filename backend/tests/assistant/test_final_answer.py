"""Final-answer hardening: tool-protocol artifacts in plain text are never accepted as an
answer and never executed. Only parsed AIMessage.tool_calls can run tools."""

import uuid

import pytest

from app.agent.assistant import AssistantError, AssistantLimits, CommerceAssistant
from app.agent.context import AgentContext
from tests.assistant.fakes import (
    DownDatabase,
    ai_text,
    ai_tools,
    call,
    make_provider,
    offline_tools,
)

CTX = AgentContext(uuid.uuid4(), "req-final")


@pytest.fixture
def db():
    return DownDatabase()


def run(db, *script):
    provider, model = make_provider(*script)
    assistant = CommerceAssistant(provider, tools=offline_tools(db), limits=AssistantLimits())
    return assistant, model


def rejected(db, text, *, name=None, reason="textual_tool_call"):
    assistant, model = run(db, ai_text(text))
    with pytest.raises(AssistantError) as exc:
        assistant.run("What is our refund policy for a shipment delayed 10 days?", CTX)
    err = exc.value
    assert err.code == "agent_protocol_error"
    assert err.tool_calls == []  # nothing executed
    assert db.sessions_opened == 0
    assert len(model.invocations) == 1  # no follow-up round with a fabricated ToolMessage
    [inv] = err.invalid_tool_calls
    assert (inv.name, inv.reason) == (name, reason)
    return err


@pytest.mark.parametrize("text", ["{}", "[]", "  {}\n", "```json\n{}\n```"])
def test_empty_json_is_not_an_answer(db, text):
    err = rejected(db, text, reason="empty_structured_output")
    assert err.detail == "protocol_artifact"


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "get_policy", "parameters": {"topic": "refund", "delay_days": 10}}',
        '{"name": "get_policy", "arguments": {"topic": "refund"}}',
        '```json\n{"name": "get_policy", "parameters": {"topic": "refund"}}\n```',
        '[{"name": "get_policy", "parameters": {}}]',
        '<tool_call>\n{"name": "get_policy", "arguments": {"topic": "refund"}}\n</tool_call>',
    ],
)
def test_textual_call_to_unregistered_tool_is_rejected(db, text):
    rejected(db, text, name="get_policy")


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "get_order", "parameters": {"order_number": "ORD-1001"}}',
        '<tool_call>{"name": "get_order", "arguments": {"order_number": "ORD-1001"}}</tool_call>',
    ],
)
def test_textual_call_to_a_registered_tool_is_rejected_not_executed(db, text):
    rejected(db, text, name="get_order")


def test_tool_call_markup_without_parseable_json_is_rejected(db):
    rejected(db, "<tool_call>get_order ORD-1001</tool_call>", name=None)


def test_parsed_tool_calls_still_execute_normally(db):
    assistant, model = run(
        db,
        ai_tools(call("get_order", "p-1", order_number="ORD-1001")),
        ai_text("The order service is unavailable right now."),
    )
    result = assistant.run("Show me order ORD-1001", CTX)
    assert [s.tool for s in result.tool_calls] == ["get_order"]
    assert db.sessions_opened == 1 and len(model.invocations) == 2


@pytest.mark.parametrize(
    "text",
    [
        "Hello! How can I help?",
        "Order ORD-1001 is delivered; total 179.89 USD.",
        'The status field reads "delivered" for ORD-1001.',
        '{"answer": "hello"} is not a tool call, it is just JSON text.',
        "I don't have access to company policies yet, so I can't state the refund policy.",
        # Scope guard: JSON with a "name" but no parameters/arguments/args is NOT a call.
        '{"name": "Ava Thompson", "status": "active"}',
    ],
)
def test_ordinary_text_remains_a_valid_answer(db, text):
    assistant, _ = run(db, ai_text(text))
    assert assistant.run("hello", CTX).answer == text


def test_rejection_is_logged_as_protocol_error(db, caplog):
    import logging

    assistant, _ = run(db, ai_text('{"name": "get_policy", "parameters": {"topic": "refund"}}'))
    with caplog.at_level(logging.INFO, logger="app.agent.assistant"), pytest.raises(AssistantError):
        assistant.run("refund policy?", CTX)
    [rec] = [r for r in caplog.records if r.name == "app.agent.assistant"]
    assert (rec.outcome, rec.detail, rec.invalid_tool_calls) == (
        "agent_protocol_error",
        "protocol_artifact",
        1,
    )
    assert "get_policy" not in caplog.text or rec.tool_names == []
