"""LangGraph assistant: topology, routing decisions, nodes, runtime context, logging.
Scripted model, real tools, deterministic "down" DB. No database or network."""

import json
import logging
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END

import app.agent.graph.runner as runner_module
from app.agent.assistant import AssistantError, AssistantLimits, CommerceAssistant
from app.agent.assistant.executor import ToolExecutionError, ToolExecutor
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant, build_commerce_graph
from app.agent.graph.builder import recursion_limit_for
from app.agent.graph.routing import (
    MODEL,
    TOOLS,
    evaluate_model_turn,
    route_after_model,
    route_after_tools,
)
from app.agent.graph.state import RUN_RESET
from app.agent.prompts import assistant as assistant_prompt
from app.agent.tools import COMMERCE_TOOL_NAMES
from tests.assistant.fakes import (
    DownDatabase,
    ai_text,
    ai_tools,
    call,
    make_provider,
    offline_tools,
)

CTX = AgentContext(uuid.uuid4(), "req-graph")
D = AssistantLimits()


@pytest.fixture
def db():
    return DownDatabase()


def build(db, *script, limits=None, **kw):
    provider, model = make_provider(*script)
    assistant = CommerceGraphAssistant(
        provider, tools=offline_tools(db), limits=limits or AssistantLimits(), **kw
    )
    return assistant, model


# --- topology ----------------------------------------------------------------------------------
def test_graph_topology():
    provider, _ = make_provider()
    g = build_commerce_graph(provider, tools=offline_tools(DownDatabase()), limits=D).get_graph()
    assert set(g.nodes) == {"__start__", MODEL, TOOLS, "__end__"}
    assert {(e.source, e.target, e.conditional) for e in g.edges} == {
        ("__start__", MODEL, False),
        (MODEL, TOOLS, True),
        (MODEL, "__end__", True),
        (TOOLS, MODEL, True),
        (TOOLS, "__end__", True),
    }


def test_graph_can_render_mermaid_text_without_extra_dependencies():
    provider, _ = make_provider()
    text = build_commerce_graph(provider, tools=offline_tools(DownDatabase()), limits=D)
    mermaid = text.get_graph().draw_mermaid()
    assert "model" in mermaid and "tools" in mermaid and "__end__" in mermaid


def test_recursion_limit_is_defensive_headroom_over_app_limits():
    for rounds in range(1, 11):
        needed = 2 * rounds - 1  # model/tools alternating, last step is a model step
        assert recursion_limit_for(AssistantLimits(max_model_rounds=rounds)) > needed


# --- evaluate_model_turn: order of checks -------------------------------------------------------
def _eval(ai, *, round_no=1, seen=(), so_far=0, limits=D):
    return evaluate_model_turn(
        ai, round_no=round_no, seen_call_ids=seen, tool_calls_so_far=so_far, limits=limits
    )


BAD = {"type": "invalid_tool_call", "id": "x", "name": "get_order", "args": "{", "error": "e"}


def test_invalid_tool_calls_win_over_parsed_calls():
    ai = AIMessage(
        content="", tool_calls=[call("get_order", "a", order_number="A")], invalid_tool_calls=[BAD]
    )
    d = _eval(ai)
    assert (d.kind, d.error_code, d.error_detail) == (
        "error",
        "agent_protocol_error",
        "invalid_tool_calls",
    )
    assert [c.reason for c in d.invalid_calls] == ["unparseable"]


@pytest.mark.parametrize("text", ["", "   ", "{}", '{"name": "get_order", "args": {}}'])
def test_tool_calling_message_may_have_any_visible_text(text):
    """The final-answer guard runs only when there are no parsed tool calls."""
    ai = AIMessage(content=text, tool_calls=[call("get_order", "a", order_number="A")])
    d = _eval(ai)
    assert (d.kind, d.new_call_ids) == ("tools", ("a",))


@pytest.mark.parametrize(
    ("text", "code", "detail"),
    [
        ("", "agent_empty_answer", None),
        ("{}", "agent_protocol_error", "protocol_artifact"),
        ("<tool_call>x</tool_call>", "agent_protocol_error", "protocol_artifact"),
    ],
)
def test_final_answer_guard_without_tool_calls(text, code, detail):
    d = _eval(ai_text(text))
    assert (d.kind, d.error_code, d.error_detail) == ("error", code, detail)


def test_final_answer_accepted():
    d = _eval(ai_text("  Order ORD-1001 is shipped. "))
    assert (d.kind, d.answer) == ("answer", "Order ORD-1001 is shipped.")


def test_id_checks_precede_limits():
    calls = [call("get_order", "dup", order_number=str(i)) for i in range(9)]
    d = _eval(ai_tools(*calls), limits=AssistantLimits(max_tool_calls_per_turn=4))
    assert (d.error_detail, d.invalid_calls[0].reason) == ("tool_call_id", "duplicate_id")


def test_seen_ids_from_earlier_rounds_are_duplicates():
    d = _eval(ai_tools(call("get_order", "old", order_number="A")), seen=("old",))
    assert d.invalid_calls[0].reason == "duplicate_id"


@pytest.mark.parametrize(
    ("n", "so_far", "round_no", "limits", "detail"),
    [
        (5, 0, 1, AssistantLimits(max_tool_calls_per_turn=4), "max_tool_calls_per_turn"),
        (3, 6, 3, AssistantLimits(max_tool_calls=8), "max_tool_calls"),
        (1, 0, 5, AssistantLimits(max_model_rounds=5), "max_model_rounds"),
        (4, 4, 2, AssistantLimits(max_tool_calls=8), None),  # exactly at the budget: allowed
    ],
)
def test_batch_limits(n, so_far, round_no, limits, detail):
    ai = ai_tools(*[call("list_delayed_shipments", f"id{i}") for i in range(n)])
    d = _eval(ai, so_far=so_far, round_no=round_no, limits=limits)
    if detail is None:
        assert d.kind == "tools"
    else:
        assert (d.kind, d.error_code, d.error_detail) == ("error", "agent_limit_exceeded", detail)


def test_route_functions_only_read_recorded_state():
    pending = ai_tools(call("get_order", "a", order_number="A"))
    assert route_after_model({"pending": pending, "error": None}) == TOOLS
    assert route_after_model({"pending": None, "error": None, "answer": "x"}) == END
    err = {"code": "agent_limit_exceeded", "message": None, "detail": "max_tool_calls"}
    assert route_after_model({"pending": pending, "error": err}) == END
    assert route_after_tools({"error": None}) == MODEL
    assert route_after_tools({"error": err}) == END


# --- nodes / runner ----------------------------------------------------------------------------
def test_model_is_bound_exactly_to_the_registry(db):
    assistant, model = build(db, ai_tools(call("get_order", order_number="ORD-1")), ai_text("x"))
    assistant.run("x", CTX)
    assert [i.tool_names for i in model.invocations] == [COMMERCE_TOOL_NAMES] * 2
    assert assistant.bound_tool_names == COMMERCE_TOOL_NAMES


def test_messages_and_tool_schemas_never_carry_tenant_context(db):
    assistant, model = build(
        db,
        ai_tools(call("get_order", order_number="ORD-1001", tenant_id=str(uuid.uuid4()))),
        ai_text("ok"),
    )
    assistant.run("Show me order ORD-1001", CTX)
    schemas = json.dumps([convert_to_openai_tool(t) for t in offline_tools(db)]).lower()
    for term in ("tenant", "runtime", "request_id", "agentcontext", "scope"):
        assert term not in schemas
    for inv in model.invocations:
        blob = json.dumps([m.model_dump() for m in inv.messages], default=str)
        assert str(CTX.tenant_id) not in blob and "req-graph" not in blob


def test_final_history_order_and_pending_cleared(db):
    first = ai_tools(
        call("get_customer", "a", customer_code="C"), call("get_order", "b", order_number="O")
    )
    final = ai_text("done")
    assistant, _ = build(db, first, final)
    values = assistant.graph.invoke(
        {**RUN_RESET, "messages": assistant_prompt.build_messages("q")},
        {"recursion_limit": 20},
        context=CTX,
    )
    kinds = [(type(m).__name__, getattr(m, "tool_call_id", None)) for m in values["messages"]]
    assert kinds == [
        ("SystemMessage", None),
        ("HumanMessage", None),
        ("AIMessage", None),
        ("ToolMessage", "a"),
        ("ToolMessage", "b"),
        ("AIMessage", None),
    ]
    assert values["messages"][2].tool_calls == first.tool_calls
    assert values["pending"] is None and values["answer"] == "done"
    assert values["seen_tool_call_ids"] == ["a", "b"] and values["model_calls"] == 2


def test_whole_batch_runs_before_the_next_model_call(db, monkeypatch):
    order: list[str] = []
    real = ToolExecutor.execute

    def spy(self, call_, context, round_no):
        order.append(f"tool:{call_['id']}")
        return real(self, call_, context, round_no)

    monkeypatch.setattr(ToolExecutor, "execute", spy)

    def second(messages):
        order.append("model:2")
        return ai_text("done")

    assistant, _ = build(
        db,
        ai_tools(*[call("list_delayed_shipments", f"t{i}") for i in range(4)]),
        second,
    )
    result = assistant.run("x", CTX)
    assert order == ["tool:t0", "tool:t1", "tool:t2", "tool:t3", "model:2"]
    assert [c.round for c in result.tool_calls] == [1, 1, 1, 1]


def test_every_tool_call_goes_through_the_tool_executor(db, monkeypatch):
    seen: list[tuple[str, AgentContext]] = []
    real = ToolExecutor.execute

    def spy(self, call_, context, round_no):
        seen.append((call_["name"], context))
        return real(self, call_, context, round_no)

    monkeypatch.setattr(ToolExecutor, "execute", spy)
    assistant, _ = build(
        db,
        ai_tools(call("get_order", "a", order_number="A"), call("run_sql", "b", query="x")),
        ai_text("done"),
    )
    result = assistant.run("x", CTX)
    assert seen == [("get_order", CTX), ("run_sql", CTX)]  # the trusted context, by identity
    assert [c.outcome for c in result.tool_calls] == ["service_unavailable", "unknown_tool"]


def test_runtime_context_must_be_agent_context(db):
    assistant, model = build(db, ai_text("x"))
    with pytest.raises(TypeError):
        assistant.run("hi", {"tenant_id": str(CTX.tenant_id)})  # type: ignore[arg-type]
    for bad in (None, {"tenant_id": str(CTX.tenant_id)}):
        with pytest.raises(TypeError):
            assistant.graph.invoke({**RUN_RESET, "messages": [HumanMessage("x")]}, context=bad)
    assert model.invocations == []


def test_tool_result_protocol_failure_ends_safely_like_step5(db, monkeypatch):
    def broken(self, call_, context, round_no):
        raise ToolExecutionError("lost id")

    monkeypatch.setattr(ToolExecutor, "execute", broken)
    outcomes = []
    for cls in (CommerceAssistant, CommerceGraphAssistant):
        provider, model = make_provider(ai_tools(call("get_order", "a", order_number="A")))
        with pytest.raises(AssistantError) as exc:
            cls(provider, tools=offline_tools(db), limits=D).run("x", CTX)
        outcomes.append(
            (exc.value.code, exc.value.detail, exc.value.model_calls, len(model.invocations))
        )
    assert outcomes[0] == outcomes[1] == ("agent_protocol_error", "tool_result", 1, 1)


def test_rejected_turn_is_not_appended_to_history(db):
    assistant, _ = build(
        db,
        ai_tools(*[call("get_order", f"i{i}", order_number="A") for i in range(5)]),
        limits=AssistantLimits(max_tool_calls_per_turn=4),
    )
    values = assistant.graph.invoke(
        {**RUN_RESET, "messages": assistant_prompt.build_messages("q")}, context=CTX
    )
    assert [type(m) for m in values["messages"]] == [SystemMessage, HumanMessage]
    assert values["pending"] is None and values["error"]["detail"] == "max_tool_calls_per_turn"
    assert db.sessions_opened == 0


def test_graph_recursion_limit_is_mapped_to_a_safe_limit_error(db, monkeypatch):
    monkeypatch.setattr(runner_module, "recursion_limit_for", lambda _limits: 2)
    endless = [ai_tools(call("list_delayed_shipments")) for _ in range(5)]
    assistant, _ = build(db, *endless)
    with pytest.raises(AssistantError) as exc:
        assistant.run("x", CTX)
    assert (exc.value.code, exc.value.detail) == ("agent_limit_exceeded", "recursion_limit")


def test_default_graph_assistant_uses_build_commerce_tools_and_settings_limits():
    provider, _ = make_provider(ai_text("x"))
    assert CommerceGraphAssistant(provider).bound_tool_names == COMMERCE_TOOL_NAMES


# --- logging ------------------------------------------------------------------------------------
USER_TEXT = "Show me order ORD-1001 PRIVATE-USER-TEXT"


def test_run_summary_log_contains_only_safe_fields(db, caplog):
    assistant, _ = build(
        db, ai_tools(call("get_order", order_number="ORD-1001")), ai_text("PRIVATE-MODEL-ANSWER")
    )
    with caplog.at_level(logging.DEBUG):
        assistant.run(USER_TEXT, CTX)
    [rec] = [r for r in caplog.records if r.name == "app.agent.graph"]
    assert (
        rec.runner,
        rec.outcome,
        rec.model_calls,
        rec.tool_call_count,
        rec.tool_names,
        rec.graph_steps,
        rec.checkpointed,
    ) == ("langgraph", "ok", 2, 1, ["get_order"], 3, False)
    assert (rec.provider, rec.model, rec.prompt_version, rec.request_id) == (
        "fake",
        "scripted",
        "commerce-assistant-v1",
        "req-graph",
    )
    assert not hasattr(rec, "thread_key")
    for secret in (
        "PRIVATE-USER-TEXT",
        "PRIVATE-MODEL-ANSWER",
        assistant_prompt.SYSTEM_PROMPT[:40],
        'service_unavailable", "message',
        "scope_digest",
    ):
        assert secret not in caplog.text


def test_failed_run_is_logged_with_code_and_detail(db, caplog):
    assistant, _ = build(db, ai_text("{}"))
    with caplog.at_level(logging.INFO), pytest.raises(AssistantError):
        assistant.run(USER_TEXT, CTX)
    [rec] = [r for r in caplog.records if r.name == "app.agent.graph"]
    assert (rec.outcome, rec.detail, rec.invalid_tool_calls) == (
        "agent_protocol_error",
        "protocol_artifact",
        1,
    )
    assert "PRIVATE-USER-TEXT" not in caplog.text


def test_tool_messages_in_state_are_step3_envelopes(db):
    assistant, model = build(db, ai_tools(call("get_order", "e", order_number="")), ai_text("x"))
    assistant.run("x", CTX)
    [tm] = [m for m in model.invocations[1].messages if isinstance(m, ToolMessage)]
    assert json.loads(tm.content)["error"]["code"] == "invalid_arguments"
