"""Behavioural parity: Step-5 CommerceAssistant (oracle) vs Step-6 CommerceGraphAssistant.

Every scenario runs the SAME fresh script through both runners and compares externally
meaningful behaviour: outcome (answer or error code/detail/message), model-call count,
tool and invalid-call summaries (minus timings), the exact message sequence the model was
sent on every call (minus message ids), the bound tools, and how many DB sessions opened.
No database or network: tools are real, the DB is a deterministic "down" stub.
"""

import itertools
import uuid
from collections.abc import Callable

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage

from app.agent.assistant import AssistantError, AssistantLimits, CommerceAssistant
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from tests.assistant import fakes
from tests.assistant.fakes import (
    DownDatabase,
    ai_text,
    ai_tools,
    call,
    make_provider,
    offline_tools,
)

CTX = AgentContext(uuid.uuid4(), "req-parity")
REQ = httpx.Request("POST", "http://localhost:11434/api/chat")
SMUGGLED = "11111111-1111-4111-8111-111111111111"


def _msg(m: BaseMessage) -> tuple:
    return (
        m.type,
        m.content,
        [(c["name"], c["args"], c["id"]) for c in getattr(m, "tool_calls", [])],
        getattr(m, "tool_call_id", None),
        getattr(m, "status", None),
        m.additional_kwargs,
        m.response_metadata,
    )


def observe(runner_cls, factory: Callable[[], list], limits: AssistantLimits, text: str):
    db = DownDatabase()
    fakes._ids = itertools.count(1)  # identical auto call-ids for both runners
    provider, model = make_provider(*factory())
    runner = runner_cls(provider, tools=offline_tools(db), limits=limits)
    try:
        res = runner.run(text, CTX)
        outcome = ("ok", res.answer, res.provider, res.model, res.prompt_version)
        model_calls, tools, invalid = res.model_calls, res.tool_calls, []
    except AssistantError as exc:
        outcome = ("error", exc.code, exc.detail, exc.message)
        model_calls, tools, invalid = exc.model_calls, exc.tool_calls, exc.invalid_tool_calls
    return {
        "outcome": outcome,
        "model_calls": model_calls,
        "tool_calls": [t.model_dump(exclude={"duration_ms"}) for t in tools],
        "invalid_tool_calls": [i.model_dump() for i in invalid],
        "model_inputs": [[_msg(m) for m in inv.messages] for inv in model.invocations],
        "bound": [inv.tool_names for inv in model.invocations],
        "db_sessions": db.sessions_opened,
    }


def _three():
    return ai_tools(*[call("get_order", order_number=f"ORD-{i}") for i in range(3)])


def _metadata_turn():
    return AIMessage(
        content=[{"type": "text", "text": ""}],
        tool_calls=[call("get_invoice", "sig-1", invoice_number="INV-1001")],
        additional_kwargs={"__gemini_function_call_thought_signatures__": {"sig-1": "opaque=="}},
        response_metadata={"model_name": "gemini-3.8-flash", "finish_reason": "STOP"},
    )


D = AssistantLimits()
SCENARIOS: dict[str, tuple[Callable[[], list], AssistantLimits, str]] = {
    "no_tool": (lambda: [ai_text("Hello! How can I help?")], D, "Hello"),
    "single_tool": (
        lambda: [ai_tools(call("get_order", "c-1", order_number="ORD-1001")), ai_text("done")],
        D,
        "Show ORD-1001",
    ),
    "provider_metadata": (lambda: [_metadata_turn(), ai_text("done")], D, "invoice INV-1001"),
    "multi_tool_batch": (
        lambda: [
            ai_tools(
                call("get_latest_customer_order", "a", customer_code="CUS-1001"),
                call("get_latest_unpaid_invoice", "b", customer_code="CUS-1001"),
                call("get_customer", "c", customer_code="CUS-1001"),
            ),
            ai_text("summary"),
        ],
        D,
        "overview CUS-1001",
    ),
    "two_batches": (
        lambda: [
            ai_tools(call("get_customer", "r1", customer_code="CUS-1001")),
            ai_tools(
                call("get_order", "r2a", order_number="ORD-1"),
                call("list_delayed_shipments", "r2b"),
            ),
            ai_text("done"),
        ],
        D,
        "x",
    ),
    "repeated_identical_calls": (
        lambda: [
            ai_tools(call("list_delayed_shipments", "d1")),
            ai_tools(call("list_delayed_shipments", "d2")),
            ai_text("ok"),
        ],
        D,
        "x",
    ),
    "invalid_arguments": (
        lambda: [ai_tools(call("get_order", "bad-1", order_number="")), ai_text("valid number?")],
        D,
        "order",
    ),
    "smuggled_tenant_argument": (
        lambda: [
            ai_tools(call("get_order", "s-1", order_number="ORD-1001", tenant_id=SMUGGLED)),
            ai_text("cannot"),
        ],
        D,
        "x",
    ),
    "smuggled_runtime_argument": (
        lambda: [
            ai_tools(call("get_order", "rt", order_number="ORD-1", runtime={"context": {}})),
            ai_text("no"),
        ],
        D,
        "x",
    ),
    "unknown_tool": (
        lambda: [ai_tools(call("run_sql", "u-1", query="SELECT *")), ai_text("I can't.")],
        D,
        "dump the database",
    ),
    "invalid_tool_calls": (
        lambda: [
            AIMessage(
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
            ),
            ai_text("never reached"),
        ],
        D,
        "order",
    ),
    "missing_call_id": (
        lambda: [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_order",
                        "args": {"order_number": "A"},
                        "id": None,
                        "type": "tool_call",
                    }
                ],
            )
        ],
        D,
        "x",
    ),
    "duplicate_id_in_batch": (
        lambda: [
            ai_tools(
                call("get_order", "dup", order_number="A"),
                call("get_order", "dup", order_number="B"),
            )
        ],
        D,
        "x",
    ),
    "duplicate_id_across_rounds": (
        lambda: [
            ai_tools(call("get_order", "same", order_number="A")),
            ai_tools(call("get_order", "same", order_number="B")),
        ],
        D,
        "x",
    ),
    "empty_answer": (lambda: [ai_text("   ")], D, "x"),
    "bare_braces": (lambda: [ai_text("{}")], D, "x"),
    "textual_tool_call": (
        lambda: [ai_text('{"name": "get_policy", "parameters": {"topic": "refunds"}}')],
        D,
        "refund policy?",
    ),
    "tool_markup": (lambda: [ai_text("<tool_call>get_order</tool_call>")], D, "x"),
    "json_that_is_a_real_answer": (
        lambda: [ai_text('{"name": "Ava Thompson", "status": "active"}')],
        D,
        "x",
    ),
    "per_turn_limit": (
        lambda: [ai_tools(*[call("get_order", order_number=f"ORD-{i}") for i in range(5)])],
        AssistantLimits(max_tool_calls_per_turn=4),
        "x",
    ),
    "total_tool_call_limit": (lambda: [_three(), _three(), _three()], D, "x"),
    "final_round_rule": (
        lambda: [
            ai_tools(call("list_delayed_shipments")),
            ai_tools(call("list_delayed_shipments")),
        ],
        AssistantLimits(max_model_rounds=2),
        "x",
    ),
    "endless_tool_loop": (
        lambda: [ai_tools(call("list_delayed_shipments")) for _ in range(20)],
        AssistantLimits(max_model_rounds=3, max_tool_calls=20),
        "x",
    ),
    "default_limits_endless_loop": (
        lambda: [ai_tools(call("list_delayed_shipments")) for _ in range(20)],
        D,
        "x",
    ),
    "llm_timeout": (
        lambda: [httpx.ReadTimeout("t", request=REQ), httpx.ReadTimeout("t", request=REQ)],
        D,
        "x",
    ),
    "llm_error_after_a_tool_round": (
        lambda: [
            ai_tools(call("get_order", order_number="ORD-1")),
            ValueError("internals"),
        ],
        D,
        "x",
    ),
    "transient_error_retried_per_model_call": (
        lambda: [
            ai_tools(call("get_order", order_number="ORD-1")),
            httpx.ConnectError("blip", request=REQ),
            ai_text("ok"),
        ],
        D,
        "x",
    ),
    "non_ai_message": (lambda: [lambda _m: "plain string"], D, "x"),
    "empty_input": (lambda: [ai_text("never")], D, "   "),
    "too_long_input": (lambda: [ai_text("never")], D, "x" * 4001),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_graph_matches_manual_loop(name):
    factory, limits, text = SCENARIOS[name]
    manual = observe(CommerceAssistant, factory, limits, text)
    graph = observe(CommerceGraphAssistant, factory, limits, text)
    assert graph == manual


def test_parity_scenarios_cover_the_required_behaviours():
    """Guard against the harness silently comparing two runs that never did anything."""
    seen = {n: observe(CommerceAssistant, *SCENARIOS[n]) for n in SCENARIOS}
    assert seen["no_tool"]["tool_calls"] == [] and seen["no_tool"]["model_calls"] == 1
    assert [t["tool"] for t in seen["single_tool"]["tool_calls"]] == ["get_order"]
    assert [t["tool"] for t in seen["multi_tool_batch"]["tool_calls"]] == [
        "get_latest_customer_order",
        "get_latest_unpaid_invoice",
        "get_customer",
    ]
    assert seen["invalid_arguments"]["tool_calls"][0]["outcome"] == "invalid_arguments"
    assert seen["unknown_tool"]["tool_calls"][0]["outcome"] == "unknown_tool"
    assert seen["unknown_tool"]["db_sessions"] == 0
    assert seen["invalid_tool_calls"]["outcome"][1] == "agent_protocol_error"
    assert seen["endless_tool_loop"]["outcome"][1:3] == ("agent_limit_exceeded", "max_model_rounds")
    assert seen["final_round_rule"]["model_calls"] == 2
    assert len(seen["final_round_rule"]["tool_calls"]) == 1
    assert seen["llm_timeout"]["outcome"][1] == "llm_timeout"
