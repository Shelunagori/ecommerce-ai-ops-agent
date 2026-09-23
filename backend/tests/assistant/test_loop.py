"""Assistant loop protocol tests with a scripted model and NO database/network."""

import json
import logging
import uuid

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_google_genai.chat_models import GoogleAuthenticationError

from app.agent.assistant import AssistantError, AssistantLimits, CommerceAssistant
from app.agent.assistant.executor import ToolExecutor
from app.agent.context import AgentContext
from app.agent.prompts import assistant as assistant_prompt
from app.agent.tools import COMMERCE_TOOL_NAMES
from app.agent.tools.invoke import make_runtime
from tests.assistant.fakes import (
    DownDatabase,
    ai_text,
    ai_tools,
    call,
    make_provider,
    offline_tools,
)

CTX = AgentContext(uuid.uuid4(), "req-loop")
REQ = httpx.Request("POST", "http://localhost:11434/api/chat")


@pytest.fixture
def db():
    return DownDatabase()


def build(db, *script, limits=None, retries=1):
    provider, model = make_provider(*script, retries=retries)
    assistant = CommerceAssistant(
        provider, tools=offline_tools(db), limits=limits or AssistantLimits()
    )
    return assistant, model


def tool_messages(invocation):
    return [m for m in invocation.messages if isinstance(m, ToolMessage)]


# --- basics ----------------------------------------
def test_no_tool_answer(db):
    assistant, model = build(db, ai_text("Hello! How can I help with orders or invoices?"))
    result = assistant.run("Hello", CTX)
    assert result.answer.startswith("Hello!")
    assert (result.model_calls, result.tool_calls) == (1, [])
    assert (result.provider, result.model) == ("fake", "scripted")
    assert result.prompt_version == assistant_prompt.PROMPT_VERSION == "commerce-assistant-v1"
    assert result.duration_ms >= 0
    first = model.invocations[0].messages
    assert (
        isinstance(first[0], SystemMessage) and first[0].content == assistant_prompt.SYSTEM_PROMPT
    )
    assert isinstance(first[1], HumanMessage) and first[1].content == "Hello"
    assert db.sessions_opened == 0


def test_model_is_bound_exactly_to_the_registry(db):
    assistant, model = build(db, ai_text("hi"))
    assistant.run("hi", CTX)
    assert model.invocations[0].tool_names == COMMERCE_TOOL_NAMES
    assert assistant.bound_tool_names == COMMERCE_TOOL_NAMES


def test_default_assistant_uses_build_commerce_tools():
    provider, _ = make_provider(ai_text("x"))
    assert (
        CommerceAssistant(provider, limits=AssistantLimits()).bound_tool_names
        == COMMERCE_TOOL_NAMES
    )


def test_bound_tool_schemas_and_messages_never_carry_tenant_context(db):
    assistant, model = build(
        db, ai_tools(call("get_order", order_number="ORD-1001")), ai_text("ok")
    )
    assistant.run("Show me order ORD-1001", CTX)
    schemas = json.dumps([convert_to_openai_tool(t) for t in assistant._executor.tools]).lower()
    for term in ("tenant", "runtime", "request_id", "agentcontext"):
        assert term not in schemas
    for inv in model.invocations:
        blob = json.dumps([m.model_dump() for m in inv.messages], default=str)
        assert str(CTX.tenant_id) not in blob and "req-loop" not in blob


# --- single / multi tool, protocol ordering ------------------------------------------------
def test_single_tool_result_is_fed_back_with_same_id(db):
    first = ai_tools(call("get_order", "c-1", order_number="ORD-1001"))
    assistant, model = build(db, first, ai_text("The data service is unavailable right now."))
    result = assistant.run("Show me order ORD-1001", CTX)
    second = model.invocations[1].messages
    assert second[2] is first  # the ORIGINAL AIMessage object
    [tm] = tool_messages(model.invocations[1])
    assert tm.tool_call_id == "c-1" and tm.name == "get_order"
    assert json.loads(tm.content)["error"]["code"] == "service_unavailable"
    [summary] = result.tool_calls
    assert (summary.tool, summary.arguments, summary.outcome, summary.round) == (
        "get_order",
        {"order_number": "ORD-1001"},
        "service_unavailable",
        1,
    )
    assert result.model_calls == 2


def test_original_ai_message_and_provider_metadata_survive(db):
    signature = {"__gemini_function_call_thought_signatures__": {"c-9": "opaque-signature=="}}
    first = AIMessage(
        content=[{"type": "text", "text": ""}],
        tool_calls=[call("get_invoice", "c-9", invoice_number="INV-1001")],
        additional_kwargs=dict(signature),
        response_metadata={"model_name": "gemini-3.8-flash", "finish_reason": "STOP"},
        id="run-abc",
    )
    assistant, model = build(db, first, ai_text("done"))
    assistant.run("invoice INV-1001", CTX)
    passed = model.invocations[1].messages[2]
    assert passed is first
    assert passed.additional_kwargs == signature
    assert passed.response_metadata["model_name"] == "gemini-3.8-flash" and passed.id == "run-abc"


def test_multiple_calls_in_one_response_are_executed_as_one_batch(db):
    first = ai_tools(
        call("get_latest_customer_order", "a", customer_code="CUS-1001"),
        call("get_latest_unpaid_invoice", "b", customer_code="CUS-1001"),
        call("get_customer", "c", customer_code="CUS-1001"),
    )
    assistant, model = build(db, first, ai_text("summary"))
    result = assistant.run("Show me CUS-1001's latest order and latest unpaid invoice.", CTX)
    assert len(model.invocations) == 2  # the model is NOT invoked between calls of a batch
    msgs = model.invocations[1].messages
    assert msgs[2] is first and [type(m).__name__ for m in msgs[3:]] == ["ToolMessage"] * 3
    assert [m.tool_call_id for m in msgs[3:]] == ["a", "b", "c"]  # same order as requested
    assert [s.tool for s in result.tool_calls] == [
        "get_latest_customer_order",
        "get_latest_unpaid_invoice",
        "get_customer",
    ]
    assert db.sessions_opened == 3  # executed sequentially, each with its own session


def test_next_invocation_sees_all_results_of_previous_batches(db):
    assistant, model = build(
        db,
        ai_tools(
            call("get_order", "r1a", order_number="ORD-1"),
            call("get_order", "r1b", order_number="ORD-2"),
        ),
        ai_tools(call("get_shipment", "r2", shipment_number="SHP-1")),
        ai_text("final"),
    )
    assistant.run("orders", CTX)
    assert [m.tool_call_id for m in tool_messages(model.invocations[1])] == ["r1a", "r1b"]
    assert [m.tool_call_id for m in tool_messages(model.invocations[2])] == ["r1a", "r1b", "r2"]


def test_repeated_identical_calls_execute_again(db):
    assistant, _ = build(
        db,
        ai_tools(call("get_order", order_number="ORD-1001")),
        ai_tools(call("get_order", order_number="ORD-1001")),
        ai_text("still unavailable"),
    )
    result = assistant.run("order", CTX)
    assert [s.tool for s in result.tool_calls] == ["get_order", "get_order"]
    assert db.sessions_opened == 2  # no silent deduplication, no host retry


# --- invalid model behaviour ----------------------------------------
def test_invalid_tool_calls_never_execute_and_never_fabricate_a_tool_message(db):
    bad = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "id": "x1",
                "name": "get_order",
                "args": "{order_number: ",
                "error": "bad json",
            }
        ],
    )
    assistant, model = build(db, bad, ai_text("never reached"))
    with pytest.raises(AssistantError) as exc:
        assistant.run("order", CTX)
    assert exc.value.code == "agent_protocol_error"
    assert len(model.invocations) == 1 and db.sessions_opened == 0
    assert exc.value.tool_calls == []
    [inv] = exc.value.invalid_tool_calls
    assert (inv.name, inv.reason) == ("get_order", "unparseable")


def test_schema_invalid_arguments_get_invalid_arguments_tool_message(db):
    first = ai_tools(call("get_order", "bad-1", order_number=""))
    assistant, model = build(db, first, ai_text("Please give a valid order number."))
    result = assistant.run("order", CTX)
    [tm] = tool_messages(model.invocations[1])
    assert tm.tool_call_id == "bad-1"
    assert json.loads(tm.content) == {
        "ok": False,
        "error": {
            "code": "invalid_arguments",
            "message": "Invalid arguments: order_number (string_too_short).",
        },
    }
    assert result.tool_calls[0].outcome == "invalid_arguments" and db.sessions_opened == 0


@pytest.mark.parametrize(
    ("args", "rejected"),
    [
        (
            {"order_number": "ORD-1001", "tenant_id": "11111111-1111-4111-8111-111111111111"},
            ["tenant_id"],
        ),
        ({"order_number": "ORD-1001", "request_id": "x"}, ["request_id"]),
        ({"query": "a", "limit": 500}, []),
    ],
)
def test_smuggled_or_out_of_bounds_arguments_are_rejected(db, args, rejected):
    name = "search_customers" if "query" in args else "get_order"
    assistant, model = build(db, ai_tools(call(name, "s-1", **args)), ai_text("cannot"))
    result = assistant.run("x", CTX)
    [tm] = tool_messages(model.invocations[1])
    assert (
        tm.tool_call_id == "s-1" and json.loads(tm.content)["error"]["code"] == "invalid_arguments"
    )
    summary = result.tool_calls[0]
    assert summary.outcome == "invalid_arguments" and summary.rejected_argument_names == rejected
    assert "tenant_id" not in summary.arguments and "request_id" not in summary.arguments
    assert db.sessions_opened == 0


def test_model_supplied_runtime_is_rejected_not_merged(db):
    first = ai_tools(
        call("get_order", "rt", order_number="ORD-1", runtime={"context": {"tenant_id": "x"}})
    )
    assistant, model = build(db, first, ai_text("no"))
    assistant.run("x", CTX)
    [tm] = tool_messages(model.invocations[1])
    assert (
        json.loads(tm.content)["error"]["code"] == "invalid_arguments" and db.sessions_opened == 0
    )


def test_unknown_tool_never_executes(db):
    first = ai_tools(call("run_sql", "u-1", query="SELECT * FROM customers"))
    assistant, model = build(db, first, ai_text("I can't do that."))
    result = assistant.run("dump the database", CTX)
    [tm] = tool_messages(model.invocations[1])
    assert tm.tool_call_id == "u-1" and tm.status == "error"
    assert json.loads(tm.content)["error"]["code"] == "unknown_tool"
    summary = result.tool_calls[0]
    assert (summary.tool, summary.outcome, summary.arguments) == ("run_sql", "unknown_tool", {})
    assert db.sessions_opened == 0


def test_unknown_tool_name_is_sanitised():
    ex = ToolExecutor(offline_tools(DownDatabase()))
    msg, summary = ex.execute(
        {"name": "../../etc/passwd; DROP", "args": {}, "id": "z", "type": "tool_call"}, CTX, 1
    )
    assert summary.tool == "<invalid>" and "passwd" not in msg.content


def test_non_object_arguments_are_invalid():
    ex = ToolExecutor(offline_tools(DownDatabase()))
    msg, summary = ex.execute(
        {"name": "get_order", "args": "ORD-1", "id": "z", "type": "tool_call"}, CTX, 1
    )
    assert summary.outcome == "invalid_arguments" and msg.tool_call_id == "z"


@pytest.mark.parametrize(
    ("calls", "reason"),
    [
        (
            [
                {
                    "name": "get_order",
                    "args": {"order_number": "ORD-1"},
                    "id": None,
                    "type": "tool_call",
                }
            ],
            "missing_id",
        ),
        (
            [
                call("get_order", "dup", order_number="A"),
                call("get_order", "dup", order_number="B"),
            ],
            "duplicate_id",
        ),
    ],
)
def test_missing_or_duplicate_call_ids_are_protocol_errors(db, calls, reason):
    assistant, _ = build(db, AIMessage(content="", tool_calls=calls))
    with pytest.raises(AssistantError) as exc:
        assistant.run("x", CTX)
    assert exc.value.code == "agent_protocol_error" and db.sessions_opened == 0
    assert exc.value.invalid_tool_calls[0].reason == reason


# --- limits ----------------------------------------
def test_per_turn_limit_rejects_the_whole_batch(db):
    batch = ai_tools(*[call("get_order", order_number=f"ORD-{i}") for i in range(5)])
    assistant, _ = build(db, batch, limits=AssistantLimits(max_tool_calls_per_turn=4))
    with pytest.raises(AssistantError) as exc:
        assistant.run("x", CTX)
    assert (exc.value.code, exc.value.detail) == ("agent_limit_exceeded", "max_tool_calls_per_turn")
    assert db.sessions_opened == 0 and exc.value.tool_calls == []


def test_total_tool_call_limit_rejects_the_overflowing_batch(db):
    three = lambda: ai_tools(*[call("get_order", order_number=f"ORD-{i}") for i in range(3)])  # noqa: E731
    assistant, _ = build(db, three(), three(), three(), limits=AssistantLimits(max_tool_calls=8))
    with pytest.raises(AssistantError) as exc:
        assistant.run("x", CTX)
    assert (exc.value.code, exc.value.detail) == ("agent_limit_exceeded", "max_tool_calls")
    assert len(exc.value.tool_calls) == 6 and db.sessions_opened == 6  # third batch never ran


def test_max_model_rounds_stops_an_endless_tool_loop(db):
    endless = [ai_tools(call("list_delayed_shipments")) for _ in range(20)]
    assistant, model = build(
        db, *endless, limits=AssistantLimits(max_model_rounds=3, max_tool_calls=20)
    )
    with pytest.raises(AssistantError) as exc:
        assistant.run("x", CTX)
    assert (exc.value.code, exc.value.detail) == ("agent_limit_exceeded", "max_model_rounds")
    assert len(model.invocations) == 3 and exc.value.model_calls == 3
    assert len(exc.value.tool_calls) == 2  # last round's batch is not executed


def test_default_limits_from_settings():
    limits = AssistantLimits.from_settings()
    assert (limits.max_model_rounds, limits.max_tool_calls, limits.max_tool_calls_per_turn) == (
        5,
        8,
        4,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_model_rounds": 11},
        {"max_tool_calls": 21},
        {"max_tool_calls_per_turn": 9},
        {"max_model_rounds": 0},
        {"max_tool_calls": True},
    ],
)
def test_limits_have_hard_caps(kwargs):
    with pytest.raises(ValueError):
        AssistantLimits(**kwargs)


def test_settings_enforce_hard_caps():
    from pydantic import ValidationError

    from tests.conftest import make_settings

    for field, value in (
        ("assistant_max_model_rounds", 11),
        ("assistant_max_tool_calls", 21),
        ("assistant_max_tool_calls_per_turn", 9),
    ):
        with pytest.raises(ValidationError):
            make_settings(**{field: value})


# --- answer contract ----------------------------------------
@pytest.mark.parametrize("final", [ai_text(""), ai_text("   \n")])
def test_empty_answer(db, final):
    assistant, _ = build(db, final)
    with pytest.raises(AssistantError) as exc:
        assistant.run("x", CTX)
    assert exc.value.code == "agent_empty_answer"


def test_reasoning_blocks_are_never_returned(db):
    final = AIMessage(
        content=[
            {"type": "thinking", "thinking": "SECRET-CHAIN-OF-THOUGHT"},
            {"type": "reasoning", "reasoning": "SECRET-REASONING"},
            {"type": "text", "text": "Order ORD-1001 is delivered."},
        ]
    )
    assistant, _ = build(db, final)
    result = assistant.run("status of ORD-1001", CTX)
    assert result.answer == "Order ORD-1001 is delivered."
    assert "SECRET" not in result.model_dump_json()


@pytest.mark.parametrize("text", ["", "   ", "x" * 4001, None])
def test_invalid_input_never_calls_the_model(db, text):
    assistant, model = build(db, ai_text("x"))
    with pytest.raises(AssistantError) as exc:
        assistant.run(text, CTX)  # type: ignore[arg-type]
    assert exc.value.code == "agent_input_invalid" and model.invocations == []


def test_context_must_be_agent_context(db):
    assistant, _ = build(db, ai_text("x"))
    with pytest.raises(TypeError):
        assistant.run("hi", {"tenant_id": str(CTX.tenant_id)})  # type: ignore[arg-type]


# --- provider errors (Step 4 mapping) -----------------------------------------------------------
@pytest.mark.parametrize(
    ("exc", "code", "attempts"),
    [
        (httpx.ReadTimeout("t", request=REQ), "llm_timeout", 2),
        (httpx.ConnectError("refused", request=REQ), "llm_unavailable", 2),
        (GoogleAuthenticationError("401 key AIzaSECRET invalid"), "llm_auth_failed", 1),
        (ValueError("boom internals"), "llm_internal_error", 1),
    ],
)
def test_provider_errors_map_to_safe_assistant_errors(db, exc, code, attempts):
    assistant, model = build(db, exc, exc)
    with pytest.raises(AssistantError) as caught:
        assistant.run("x", CTX)
    assert caught.value.code == code and len(model.invocations) == attempts
    assert "AIzaSECRET" not in str(caught.value) and "internals" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_non_ai_message_from_provider_is_output_error(db):
    assistant, _ = build(db, lambda _m: "plain string")  # type: ignore[arg-type,return-value]
    with pytest.raises(AssistantError) as exc:
        assistant.run("x", CTX)
    assert exc.value.code == "llm_output_invalid"


def test_provider_retry_is_per_model_call_not_per_tool(db):
    assistant, model = build(
        db,
        ai_tools(call("get_order", order_number="ORD-1")),
        httpx.ConnectError("blip", request=REQ),  # transient on the 2nd model call
        ai_text("ok"),
    )
    result = assistant.run("x", CTX)
    assert len(model.invocations) == 3 and result.model_calls == 2
    assert db.sessions_opened == 1  # the tool was NOT re-run by the retry


# --- logging ----------------------------------------
USER_TEXT = "Show me order ORD-1001 PRIVATE-USER-TEXT"


def test_run_summary_log_contains_only_safe_fields(db, caplog):
    assistant, _ = build(
        db, ai_tools(call("get_order", order_number="ORD-1001")), ai_text("PRIVATE-MODEL-ANSWER")
    )
    with caplog.at_level(logging.DEBUG):
        assistant.run(USER_TEXT, CTX)
    [rec] = [r for r in caplog.records if r.name == "app.agent.assistant"]
    assert (rec.outcome, rec.model_calls, rec.tool_call_count, rec.tool_names) == (
        "ok",
        2,
        1,
        ["get_order"],
    )
    assert (rec.provider, rec.model, rec.prompt_version, rec.request_id) == (
        "fake",
        "scripted",
        "commerce-assistant-v1",
        "req-loop",
    )
    for secret in (
        "PRIVATE-USER-TEXT",
        "PRIVATE-MODEL-ANSWER",
        assistant_prompt.SYSTEM_PROMPT[:40],
        'service_unavailable", "message',
    ):
        assert secret not in caplog.text


def test_failed_run_is_logged_with_code(db, caplog):
    assistant, _ = build(db, ai_text(""))
    with caplog.at_level(logging.INFO), pytest.raises(AssistantError):
        assistant.run(USER_TEXT, CTX)
    [rec] = [r for r in caplog.records if r.name == "app.agent.assistant"]
    assert rec.outcome == "agent_empty_answer" and "PRIVATE-USER-TEXT" not in caplog.text


def test_executor_uses_the_same_runtime_mechanism_as_step_3():
    """The executor injects make_runtime(context) - the Step 3 ToolRuntime path."""
    tools = offline_tools(DownDatabase())
    direct = tools[2].invoke({"order_number": "ORD-1", "runtime": make_runtime(CTX)})
    msg, _ = ToolExecutor(tools).execute(call("get_order", "q", order_number="ORD-1"), CTX, 1)
    assert json.loads(msg.content) == direct
