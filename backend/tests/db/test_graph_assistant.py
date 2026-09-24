"""LangGraph path end-to-end on real PostgreSQL. Only the model is fake (scripted):

    graph -> MODEL -> route -> TOOLS -> ToolExecutor -> Step-3 tool -> service -> PostgreSQL
          -> ToolMessage -> MODEL -> END

Runs are checkpointed (InMemorySaver) so the full graph message history can be inspected
for cross-tenant data, and one-shot runs are compared with the Step-5 oracle.
"""

import json
import logging
import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.orm import sessionmaker

import app.db.session as db_session_module
from app.agent.assistant import AssistantLimits, CommerceAssistant
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.profile import STEP5_PARITY_PROFILE
from app.agent.tools import ToolDependencies, build_commerce_tools
from scripts import run_graph_assistant
from tests.assistant.fakes import ai_text, ai_tools, call, make_provider
from tests.db.conftest import fixed_clock


@pytest.fixture
def real_tools(db_engine, monkeypatch):
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    return build_commerce_tools(
        ToolDependencies(session_scope=db_session_module.read_only_session, clock=fixed_clock)
    )


@pytest.fixture
def ctx_a(tenant_a):
    return AgentContext(tenant_a.tenant_id, "req-a")


@pytest.fixture
def ctx_b(tenant_b):
    return AgentContext(tenant_b.tenant_id, "req-b")


def answer_from_results(messages):
    """Fake 'final answer': echo the tool envelopes it received (proves grounding input)."""
    results = [json.loads(m.content) for m in messages if isinstance(m, ToolMessage)]
    return AIMessage(content=json.dumps(results))


def run(real_tools, ctx, *script, text="question", saver=None, thread="t1"):
    saver = saver or InMemorySaver()
    provider, model = make_provider(*script)
    assistant = CommerceGraphAssistant(
        provider, tools=real_tools, limits=AssistantLimits(), checkpointer=saver
    )
    result = assistant.run(text, ctx, thread_id=thread)
    history = assistant.graph.get_state(assistant.thread_config(ctx, thread)).values["messages"]
    return result, model, json.loads(result.answer), history


def history_blob(history):
    return json.dumps([m.model_dump() for m in history], default=str)


def test_same_order_number_per_tenant(real_tools, ctx_a, ctx_b):
    script = lambda: (ai_tools(call("get_order", order_number="ORD-1001")), answer_from_results)  # noqa: E731
    ra, _, [a], ha = run(real_tools, ctx_a, *script(), text="Show me order ORD-1001")
    rb, _, [b], hb = run(real_tools, ctx_b, *script(), text="Show me order ORD-1001")
    assert (a["data"]["currency"], a["data"]["total_amount"]) == ("USD", "179.89")
    assert (b["data"]["currency"], b["data"]["total_amount"]) == ("EUR", "248.00")
    for value in ("EUR", "Linen Bedding", "248.00"):
        assert value not in history_blob(ha)  # no BluePeak data in Northstar's graph history
    for value in ("USD", "Wireless Earbuds", "179.89"):
        assert value not in history_blob(hb)
    for ctx, history in ((ctx_a, ha), (ctx_b, hb)):
        assert str(ctx.tenant_id) not in history_blob(history)
    assert ra.tool_calls[0].outcome == rb.tool_calls[0].outcome == "success"


def test_same_thread_name_in_two_tenants_keeps_data_apart(real_tools, ctx_a, ctx_b):
    saver = InMemorySaver()
    script = lambda: (ai_tools(call("get_order", order_number="ORD-1001")), answer_from_results)  # noqa: E731
    _, _, _, ha = run(real_tools, ctx_a, *script(), saver=saver, thread="shared")
    _, model_b, [b], hb = run(real_tools, ctx_b, *script(), saver=saver, thread="shared")
    assert b["data"]["currency"] == "EUR"
    sent_to_b = json.dumps(
        [m.model_dump() for i in model_b.invocations for m in i.messages], default=str
    )
    for value in ("USD", "Wireless Earbuds", "179.89"):
        assert value not in sent_to_b and value not in history_blob(hb)
    assert "USD" in history_blob(ha)  # A's thread is intact and separate


def test_repeated_product_order_keeps_both_lines(real_tools, ctx_a):
    _, _, [r], _ = run(
        real_tools, ctx_a, ai_tools(call("get_order", order_number="ORD-1010")), answer_from_results
    )
    assert [(i["sku"], i["quantity"], i["unit_price"]) for i in r["data"]["items"]] == [
        ("SKU-1005", 4, "24.95"),
        ("SKU-1005", 2, "19.96"),
    ]


def test_latest_unpaid_invoice(real_tools, ctx_a):
    result, _, [r], _ = run(
        real_tools,
        ctx_a,
        ai_tools(call("get_latest_unpaid_invoice", customer_code="CUS-1001")),
        answer_from_results,
    )
    assert (r["data"]["invoice_number"], r["data"]["amount"], r["data"]["is_overdue"]) == (
        "INV-1004",
        "223.95",
        False,
    )
    assert result.tool_calls[0].arguments == {"customer_code": "CUS-1001"}


def test_delayed_shipments(real_tools, ctx_a, ctx_b):
    _, _, [a], _ = run(
        real_tools, ctx_a, ai_tools(call("list_delayed_shipments")), answer_from_results
    )
    _, _, [b], _ = run(
        real_tools, ctx_b, ai_tools(call("list_delayed_shipments")), answer_from_results
    )
    assert [s["shipment_number"] for s in a["data"]["items"]] == ["SHP-1003"]
    assert [(s["shipment_number"], s["delay_reason"]) for s in b["data"]["items"]] == [
        ("SHP-1002", None)
    ]


def test_multi_tool_batch_against_database(real_tools, ctx_a):
    result, model, results, history = run(
        real_tools,
        ctx_a,
        ai_tools(
            call("get_latest_customer_order", "o", customer_code="CUS-1001"),
            call("get_latest_unpaid_invoice", "i", customer_code="CUS-1001"),
        ),
        answer_from_results,
    )
    assert [r["data"].get("invoice_number") or r["data"]["order_number"] for r in results] == [
        "ORD-1004",
        "INV-1004",
    ]
    assert [m.tool_call_id for m in history if isinstance(m, ToolMessage)] == ["o", "i"]
    assert len(model.invocations) == 2


@pytest.mark.parametrize(
    ("tool", "arg", "ref", "code"),
    [
        ("get_order", "order_number", "ORD-1006", "order_not_found"),
        ("get_customer", "customer_code", "CUS-1005", "customer_not_found"),
        ("get_invoice", "invoice_number", "INV-1005", "invoice_not_found"),
        ("get_shipment", "shipment_number", "SHP-1005", "shipment_not_found"),
        ("get_product", "sku", "SKU-1005", "product_not_found"),
    ],
)
def test_other_tenants_reference_looks_like_plain_not_found(
    real_tools, ctx_b, tool, arg, ref, code
):
    """Northstar-only reference, asked in BluePeak's context."""
    cross, _, [c], _ = run(
        real_tools, ctx_b, ai_tools(call(tool, **{arg: ref})), answer_from_results
    )
    _, _, [m], _ = run(
        real_tools, ctx_b, ai_tools(call(tool, **{arg: "ZZZ-9999"})), answer_from_results
    )
    assert c["error"]["code"] == code
    assert json.dumps(c).replace(ref, "<R>") == json.dumps(m).replace("ZZZ-9999", "<R>")
    assert cross.tool_calls[0].outcome == "not_found"


def test_smuggled_tenant_cannot_switch_tenants(real_tools, ctx_a, tenant_b):
    result, _, [r], history = run(
        real_tools,
        ctx_a,
        ai_tools(call("get_order", order_number="ORD-1001", tenant_id=str(tenant_b.tenant_id))),
        answer_from_results,
    )
    assert r["error"]["code"] == "invalid_arguments"
    tool_results = [m.content for m in history if isinstance(m, ToolMessage)]
    assert "EUR" not in json.dumps(tool_results) and "Linen Bedding" not in json.dumps(tool_results)
    assert result.tool_calls[0].rejected_argument_names == ["tenant_id"]


def test_every_executed_call_goes_through_the_step3_tool(real_tools, ctx_a, caplog):
    with caplog.at_level(logging.INFO, logger="app.agent.tools"):
        run(
            real_tools,
            ctx_a,
            ai_tools(
                call("get_order", order_number="ORD-1001"), call("get_product", sku="SKU-1001")
            ),
            answer_from_results,
        )
    records = [r for r in caplog.records if r.name == "app.agent.tools"]
    assert [(r.tool, r.outcome, r.tenant_id, r.request_id) for r in records] == [
        ("get_order", "ok", str(ctx_a.tenant_id), "req-a"),
        ("get_product", "ok", str(ctx_a.tenant_id), "req-a"),
    ]


@pytest.mark.parametrize(
    "script",
    [
        lambda: [ai_tools(call("get_order", "p1", order_number="ORD-1010")), answer_from_results],
        lambda: [
            ai_tools(
                call("get_customer", "p2", customer_code="CUS-1001"),
                call("get_invoice", "p3", invoice_number="INV-1005"),
            ),
            ai_tools(call("list_delayed_shipments", "p4")),
            answer_from_results,
        ],
    ],
    ids=["repeated_lines", "two_rounds_with_not_found"],
)
def test_parity_with_step5_on_real_data(real_tools, ctx_a, script):
    outs = []
    for cls in (CommerceAssistant, CommerceGraphAssistant):
        provider, model = make_provider(*script())
        extra = {"profile": STEP5_PARITY_PROFILE} if cls is CommerceGraphAssistant else {}
        result = cls(provider, tools=real_tools, limits=AssistantLimits(), **extra).run("q", ctx_a)
        outs.append(
            (
                result.answer,
                result.model_calls,
                [c.model_dump(exclude={"duration_ms"}) for c in result.tool_calls],
                [[(m.type, m.content) for m in i.messages] for i in model.invocations],
            )
        )
    assert outs[0] == outs[1]


# --- CLI ----------------------------------------------------------------------------------------
@pytest.fixture
def cli(db_engine, monkeypatch):
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    return run_graph_assistant


RESULT_KEYS = {
    "answer",
    "provider",
    "model",
    "prompt_version",
    "model_calls",
    "tool_calls",
    "duration_ms",
    "retrievals",  # Step 9 (empty for commerce-only questions)
    "citations",
}


def test_cli_runs_with_fake_provider(cli, tenant_a, monkeypatch, capsys):
    provider, _ = make_provider(
        ai_tools(call("get_order", order_number="ORD-1001")),
        ai_text("ORD-1001 was delivered; total 179.89 USD."),
    )
    monkeypatch.setattr(cli, "get_llm_provider", lambda **_kw: provider)
    code = cli.main(["--tenant", str(tenant_a.tenant_id), "--text", "Show me order ORD-1001"])
    out = capsys.readouterr()
    assert code == 0
    result = json.loads(out.out)
    assert set(result) == RESULT_KEYS and result["model_calls"] == 2
    assert result["tool_calls"][0]["tool"] == "get_order"
    assert str(tenant_a.tenant_id) not in out.out and "in-memory" not in out.err


def test_cli_thread_continues_within_one_process(cli, tenant_a, monkeypatch, capsys):
    provider, model = make_provider(ai_text("first"), ai_text("second"))
    monkeypatch.setattr(cli, "get_llm_provider", lambda **_kw: provider)
    code = cli.main(
        [
            "--tenant",
            str(tenant_a.tenant_id),
            "--thread-id",
            "demo",
            "--text",
            "one",
            "--text",
            "two",
        ]
    )
    out = capsys.readouterr()
    assert code == 0
    assert [r["answer"] for r in json.loads(out.out)] == ["first", "second"]
    assert [m.content for m in model.invocations[1].messages][1:] == ["one", "first", "two"]
    assert "lost when it exits" in out.err
    assert "demo" not in out.out


def test_cli_repeated_text_requires_thread_id(cli, tenant_a):
    with pytest.raises(SystemExit):
        cli.main(["--tenant", str(tenant_a.tenant_id), "--text", "a", "--text", "b"])


def test_cli_rejects_unknown_tenant_before_building_a_model(cli, monkeypatch, capsys):
    built = []
    monkeypatch.setattr(cli, "get_llm_provider", lambda **kw: built.append(kw))
    code = cli.main(["--tenant", str(uuid.uuid4()), "--text", "hi"])
    assert code == 1 and built == []
    last = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert last["error"]["code"] == "tenant_not_found"


def test_cli_reports_assistant_errors_safely(cli, tenant_a, monkeypatch, capsys):
    provider, _ = make_provider(ai_text("{}"))
    monkeypatch.setattr(cli, "get_llm_provider", lambda **_kw: provider)
    code = cli.main(["--tenant", str(tenant_a.tenant_id), "--text", "hi"])
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert code == 1 and err["error"]["code"] == "agent_protocol_error"
    assert (err["turn"], err["model_calls"]) == (1, 1)
    assert err["invalid_tool_calls"][0]["reason"] == "empty_structured_output"
