"""Assistant loop -> real executor -> real Step 3 tools -> services -> PostgreSQL.
Only the model is fake (scripted)."""

import json
import logging
import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy.orm import sessionmaker

import app.db.session as db_session_module
from app.agent.assistant import AssistantLimits, CommerceAssistant
from app.agent.context import AgentContext
from app.agent.tools import ToolDependencies, build_commerce_tools
from scripts import run_assistant
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


def run(real_tools, ctx, *script, text="question"):
    provider, model = make_provider(*script)
    assistant = CommerceAssistant(provider, tools=real_tools, limits=AssistantLimits())
    result = assistant.run(text, ctx)
    return result, model, json.loads(result.answer)


def history_blob(model):
    return json.dumps(
        [m.content for inv in model.invocations for m in inv.messages if isinstance(m, ToolMessage)]
    )


def test_same_order_number_per_tenant(real_tools, ctx_a, ctx_b):
    ra, ma, [a] = run(
        real_tools,
        ctx_a,
        ai_tools(call("get_order", order_number="ORD-1001")),
        answer_from_results,
        text="Show me order ORD-1001",
    )
    rb, mb, [b] = run(
        real_tools,
        ctx_b,
        ai_tools(call("get_order", order_number="ORD-1001")),
        answer_from_results,
        text="Show me order ORD-1001",
    )
    assert (a["data"]["currency"], a["data"]["total_amount"]) == ("USD", "179.89")
    assert (b["data"]["currency"], b["data"]["total_amount"]) == ("EUR", "248.00")
    for value in ("EUR", "Linen Bedding", "248.00"):
        assert value not in history_blob(ma)  # no BluePeak data in Northstar's history
    for value in ("USD", "Wireless Earbuds", "179.89"):
        assert value not in history_blob(mb)
    assert ra.tool_calls[0].outcome == rb.tool_calls[0].outcome == "success"


def test_repeated_product_order_keeps_both_lines(real_tools, ctx_a):
    _, _, [r] = run(
        real_tools, ctx_a, ai_tools(call("get_order", order_number="ORD-1010")), answer_from_results
    )
    assert [(i["sku"], i["quantity"], i["unit_price"]) for i in r["data"]["items"]] == [
        ("SKU-1005", 4, "24.95"),
        ("SKU-1005", 2, "19.96"),
    ]


def test_latest_unpaid_invoice(real_tools, ctx_a):
    result, _, [r] = run(
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
    _, _, [a] = run(
        real_tools, ctx_a, ai_tools(call("list_delayed_shipments")), answer_from_results
    )
    _, _, [b] = run(
        real_tools, ctx_b, ai_tools(call("list_delayed_shipments")), answer_from_results
    )
    assert [s["shipment_number"] for s in a["data"]["items"]] == ["SHP-1003"]
    assert [(s["shipment_number"], s["delay_reason"]) for s in b["data"]["items"]] == [
        ("SHP-1002", None)
    ]


def test_multi_tool_batch_against_database(real_tools, ctx_a):
    result, model, results = run(
        real_tools,
        ctx_a,
        ai_tools(
            call("get_latest_customer_order", "o", customer_code="CUS-1001"),
            call("get_latest_unpaid_invoice", "i", customer_code="CUS-1001"),
        ),
        answer_from_results,
        text="Show me CUS-1001's latest order and latest unpaid invoice.",
    )
    assert [r["data"].get("invoice_number") or r["data"]["order_number"] for r in results] == [
        "ORD-1004",
        "INV-1004",
    ]
    assert [s.outcome for s in result.tool_calls] == ["success", "success"]
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
    cross, _, [c] = run(real_tools, ctx_b, ai_tools(call(tool, **{arg: ref})), answer_from_results)
    _, _, [m] = run(
        real_tools, ctx_b, ai_tools(call(tool, **{arg: "ZZZ-9999"})), answer_from_results
    )
    assert c["error"]["code"] == code
    assert json.dumps(c).replace(ref, "<R>") == json.dumps(m).replace("ZZZ-9999", "<R>")
    assert cross.tool_calls[0].outcome == "not_found"


def test_not_found_is_not_retried_by_the_host(real_tools, ctx_a, caplog):
    with caplog.at_level(logging.INFO, logger="app.agent.tools"):
        result, _, _ = run(
            real_tools,
            ctx_a,
            ai_tools(call("get_order", order_number="ORD-9999")),
            answer_from_results,
        )
    tool_logs = [r for r in caplog.records if r.name == "app.agent.tools"]
    assert len(tool_logs) == 1 and tool_logs[0].outcome == "not_found"
    assert len(result.tool_calls) == 1


def test_every_executed_call_goes_through_the_step3_tool(real_tools, ctx_a, caplog):
    """Each executed call emits exactly one Step 3 tool log line with the trusted context."""
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


def test_smuggled_tenant_cannot_switch_tenants(real_tools, ctx_a, tenant_b):
    result, model, [r] = run(
        real_tools,
        ctx_a,
        ai_tools(call("get_order", order_number="ORD-1001", tenant_id=str(tenant_b.tenant_id))),
        answer_from_results,
    )
    assert r["error"]["code"] == "invalid_arguments"
    assert "EUR" not in history_blob(model)
    assert result.tool_calls[0].rejected_argument_names == ["tenant_id"]


# --- CLI ----------------------------------------
@pytest.fixture
def cli(db_engine, monkeypatch):
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    return run_assistant


def test_cli_runs_with_fake_provider(cli, tenant_a, monkeypatch, capsys):
    provider, _ = make_provider(
        ai_tools(call("get_order", order_number="ORD-1001")),
        ai_text("ORD-1001 was delivered; total 179.89 USD."),
    )
    monkeypatch.setattr(cli, "get_llm_provider", lambda **_kw: provider)
    code = cli.main(
        [
            "--tenant",
            str(tenant_a.tenant_id),
            "--text",
            "Show me order ORD-1001",
            "--request-id",
            "cli-1",
        ]
    )
    out = capsys.readouterr()
    assert code == 0
    result = json.loads(out.out)
    assert result["answer"].startswith("ORD-1001 was delivered")
    assert result["tool_calls"][0]["tool"] == "get_order" and result["model_calls"] == 2
    assert set(result) == {
        "answer",
        "provider",
        "model",
        "prompt_version",
        "model_calls",
        "tool_calls",
        "duration_ms",
    }
    assert str(tenant_a.tenant_id) not in out.out


def test_cli_rejects_unknown_tenant_before_building_a_model(cli, monkeypatch, capsys):
    built = []
    monkeypatch.setattr(cli, "get_llm_provider", lambda **kw: built.append(kw))
    code = cli.main(["--tenant", str(uuid.uuid4()), "--text", "hi"])
    assert code == 1 and built == []
    assert (
        json.loads(capsys.readouterr().err.strip().splitlines()[-1])["error"]["code"]
        == "tenant_not_found"
    )


def test_cli_reports_assistant_errors_safely(cli, tenant_a, monkeypatch, capsys):
    provider, _ = make_provider(ai_text(""))
    monkeypatch.setattr(cli, "get_llm_provider", lambda **_kw: provider)
    code = cli.main(["--tenant", str(tenant_a.tenant_id), "--text", "hi"])
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert code == 1 and err["error"]["code"] == "agent_empty_answer" and err["model_calls"] == 1
