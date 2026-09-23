"""Agent tool boundary tests that need no database: registry, model-visible schemas,
argument validation, runtime-context handling and error classification."""

import json
import logging
import re
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
from langchain.tools import ToolRuntime
from langchain_core.utils.function_calling import convert_to_openai_tool
from sqlalchemy.exc import OperationalError

from app.agent.context import AgentContext
from app.agent.tools import COMMERCE_TOOL_NAMES, ToolDependencies, build_commerce_tools
from app.agent.tools.invoke import invoke_tool, make_runtime
from app.agent.tools.schemas import MAX_LIMIT
from app.db.session import DatabaseNotConfiguredError

EXPECTED_TOOLS = [
    "get_customer",
    "search_customers",
    "get_order",
    "list_customer_orders",
    "get_latest_customer_order",
    "get_invoice",
    "get_latest_unpaid_invoice",
    "get_shipment",
    "get_order_shipments",
    "list_delayed_shipments",
    "get_product",
    "search_products",
]

# Exactly the business arguments each tool exposes to a model.
EXPECTED_ARGS = {
    "get_customer": {"customer_code"},
    "search_customers": {"query", "limit"},
    "get_order": {"order_number"},
    "list_customer_orders": {"customer_code", "limit"},
    "get_latest_customer_order": {"customer_code"},
    "get_invoice": {"invoice_number"},
    "get_latest_unpaid_invoice": {"customer_code"},
    "get_shipment": {"shipment_number"},
    "get_order_shipments": {"order_number"},
    "list_delayed_shipments": {"limit"},
    "get_product": {"sku"},
    "search_products": {"query", "limit"},
}

VALID_ARGS = {
    "get_customer": {"customer_code": "CUS-1001"},
    "search_customers": {"query": "ava"},
    "get_order": {"order_number": "ORD-1001"},
    "list_customer_orders": {"customer_code": "CUS-1001"},
    "get_latest_customer_order": {"customer_code": "CUS-1001"},
    "get_invoice": {"invoice_number": "INV-1001"},
    "get_latest_unpaid_invoice": {"customer_code": "CUS-1001"},
    "get_shipment": {"shipment_number": "SHP-1001"},
    "get_order_shipments": {"order_number": "ORD-1001"},
    "list_delayed_shipments": {},
    "get_product": {"sku": "SKU-1001"},
    "search_products": {"query": "watch"},
}

FORBIDDEN_TERMS = ("tenant", "request_id", "runtime", "agentcontext", "toolruntime", "context")

CTX = AgentContext(uuid.uuid4(), "req-test")


@pytest.fixture(scope="module")
def tools():
    return {t.name: t for t in build_commerce_tools()}


def _model_schema(tool) -> dict:
    return tool.tool_call_schema.model_json_schema()


def _all_keys_and_strings(node) -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.append(str(k))
            out.extend(_all_keys_and_strings(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(_all_keys_and_strings(v))
    elif isinstance(node, str):
        out.append(node)
    return out


# --- registry --------------------------------------------------------------------------
def test_registry_exposes_exactly_the_intended_tools():
    names = [t.name for t in build_commerce_tools()]
    assert names == EXPECTED_TOOLS
    assert list(COMMERCE_TOOL_NAMES) == EXPECTED_TOOLS
    assert len(set(names)) == len(names)


def test_every_tool_is_marked_read_only(tools):
    for tool in tools.values():
        assert tool.metadata["read_only"] is True
        assert "read_only" in tool.tags


def test_no_generic_database_or_http_tools(tools):
    banned = re.compile(r"sql|query_database|execute|raw|table|http|request|fetch_url|api_call")
    assert not [name for name in tools if banned.search(name)]


def test_tool_names_are_snake_case_and_described(tools):
    for name, tool in tools.items():
        assert re.fullmatch(r"[a-z]+(_[a-z]+)*", name)
        assert 60 <= len(tool.description) <= 400, name
        assert "sqlalchemy" not in tool.description.lower()


def test_application_code_does_not_import_langgraph():
    app_dir = Path(__file__).resolve().parents[1] / "app"
    offenders = [
        str(p.relative_to(app_dir))
        for p in app_dir.rglob("*.py")
        if re.search(r"^\s*(from|import)\s+langgraph", p.read_text(), re.M)
    ]
    assert offenders == []


# --- model-visible schemas -----------------------------------------------------------------
@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_model_schema_contains_only_business_arguments(tools, name):
    schema = _model_schema(tools[name])
    assert set(schema.get("properties", {})) == EXPECTED_ARGS[name]
    assert "$defs" not in schema and "definitions" not in schema  # no nested models at all


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_no_tenant_request_or_runtime_anywhere_in_model_schemas(tools, name):
    """Check both LangChain's schema and the provider-format function definition, deeply."""
    tool = tools[name]
    for schema in (_model_schema(tool), convert_to_openai_tool(tool)["function"]["parameters"]):
        blob = " ".join(_all_keys_and_strings(schema)).lower()
        for term in FORBIDDEN_TERMS:
            assert term not in blob, f"{name}: '{term}' leaked into model-visible schema"


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_every_argument_is_described(tools, name):
    for arg, spec in _model_schema(tools[name]).get("properties", {}).items():
        assert spec.get("description"), f"{name}.{arg} has no description"


@pytest.mark.parametrize("name", [n for n, a in EXPECTED_ARGS.items() if "limit" in a])
def test_limits_are_bounded(tools, name):
    limit = _model_schema(tools[name])["properties"]["limit"]
    assert (limit["minimum"], limit["maximum"], limit["default"]) == (1, MAX_LIMIT, 5)
    assert MAX_LIMIT <= 20


def test_identifiers_are_length_bounded(tools):
    spec = _model_schema(tools["get_order"])["properties"]["order_number"]
    assert (spec["minLength"], spec["maxLength"]) == (1, 32)


# --- argument validation (runs before any database access) ---------------------------------
@contextmanager
def _no_db():
    raise AssertionError("database must not be touched for invalid arguments")
    yield  # pragma: no cover


@pytest.fixture(scope="module")
def guarded_tools():
    return {t.name: t for t in build_commerce_tools(ToolDependencies(session_scope=_no_db))}


@pytest.mark.parametrize(
    ("name", "args", "field"),
    [
        ("get_order", {}, "order_number"),
        ("get_order", {"order_number": ""}, "order_number"),
        ("get_order", {"order_number": "   "}, "order_number"),
        ("get_order", {"order_number": "X" * 33}, "order_number"),
        ("get_order", {"order_number": "ORD 1; DROP TABLE"}, "order_number"),
        ("get_customer", {"customer_code": 1001}, "customer_code"),
        ("search_customers", {"query": ""}, "query"),
        ("search_customers", {"query": "a" * 101}, "query"),
        ("search_customers", {"query": "a", "limit": 0}, "limit"),
        ("search_customers", {"query": "a", "limit": MAX_LIMIT + 1}, "limit"),
        ("search_products", {"query": "a", "limit": 10_000}, "limit"),
        ("list_delayed_shipments", {"limit": "all"}, "limit"),
        ("list_customer_orders", {"customer_code": "CUS-1001", "offset": 5}, "offset"),
    ],
)
def test_invalid_arguments_are_rejected(guarded_tools, name, args, field):
    result = invoke_tool(guarded_tools[name], args, CTX)
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"
    assert field in result["error"]["message"]
    assert "DROP TABLE" not in result["error"]["message"]  # input is never echoed


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
@pytest.mark.parametrize("smuggled", ["tenant_id", "request_id", "tenant"])
def test_smuggled_context_fields_are_rejected(guarded_tools, name, smuggled):
    args = {**VALID_ARGS[name], smuggled: str(uuid.uuid4())}
    result = invoke_tool(guarded_tools[name], args, CTX)
    assert result["error"]["code"] == "invalid_arguments"
    assert f"{smuggled} (extra_forbidden)" in result["error"]["message"]


def test_smuggled_tenant_is_rejected_even_via_raw_langchain_invoke(guarded_tools):
    """Without our helper: the schema itself (extra='forbid') rejects the field."""
    raw = guarded_tools["get_order"].invoke(
        {"order_number": "ORD-1001", "tenant_id": str(uuid.uuid4()), "runtime": make_runtime(CTX)}
    )
    assert json.loads(raw)["error"]["code"] == "invalid_arguments"


def test_runtime_cannot_be_supplied_as_an_argument(guarded_tools):
    forged = {
        "state": {},
        "context": {"tenant_id": str(uuid.uuid4())},
        "config": {},
        "stream_writer": None,
        "tool_call_id": None,
        "store": None,
    }
    assert (
        invoke_tool(guarded_tools["get_order"], {"order_number": "ORD-1", "runtime": forged}, CTX)[
            "error"
        ]["code"]
        == "invalid_arguments"
    )
    raw = guarded_tools["get_order"].invoke({"order_number": "ORD-1", "runtime": forged})
    assert json.loads(raw)["error"]["code"] == "internal_error"  # not a real ToolRuntime


def test_validation_failures_keep_the_envelope_at_the_framework_boundary(guarded_tools):
    """LangChain's validation callback must return str: it is the same envelope, as JSON."""
    raw = guarded_tools["get_order"].invoke({"order_number": "", "runtime": make_runtime(CTX)})
    assert isinstance(raw, str)
    assert json.loads(raw) == {
        "ok": False,
        "error": {
            "code": "invalid_arguments",
            "message": "Invalid arguments: order_number (string_too_short).",
        },
    }


def test_validation_error_as_model_tool_message(guarded_tools):
    message = guarded_tools["search_customers"].invoke(
        {
            "type": "tool_call",
            "id": "call-1",
            "name": "search_customers",
            "args": {"query": "a", "limit": 99, "runtime": make_runtime(CTX)},
        }
    )
    assert message.status == "error" and message.tool_call_id == "call-1"
    assert json.loads(message.content)["error"]["code"] == "invalid_arguments"


def test_missing_runtime_is_an_internal_error(guarded_tools):
    raw = guarded_tools["get_order"].invoke({"order_number": "ORD-1001"})
    assert json.loads(raw) == {
        "ok": False,
        "error": {"code": "internal_error", "message": "The tool failed unexpectedly."},
    }


@pytest.mark.filterwarnings("ignore::UserWarning")  # the deliberately wrong context type
def test_runtime_with_wrong_context_type_is_refused(guarded_tools):
    runtime = ToolRuntime(
        state={},
        context={"tenant_id": str(uuid.uuid4())},
        config={},
        stream_writer=None,
        tool_call_id=None,
        store=None,
    )
    result = guarded_tools["get_order"].invoke({"order_number": "ORD-1001", "runtime": runtime})
    assert isinstance(result, dict) and result["error"]["code"] == "internal_error"


# --- agent context -----------------------------------------------------------------------------
def test_agent_context_validation():
    with pytest.raises(TypeError):
        AgentContext("17243d88-ed66-5445-955b-7d7572094122")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AgentContext(uuid.uuid4(), request_id="bad id; drop")
    assert AgentContext(uuid.uuid4()).request_id is None


# --- infrastructure / internal failures -------------------------------------------------------
SECRET = "postgresql://admin:hunter2@db.internal:5432/prod"


def _deps_raising(exc: Exception) -> ToolDependencies:
    @contextmanager
    def scope():
        raise exc
        yield  # pragma: no cover

    return ToolDependencies(session_scope=scope)


class _ExplodingSession:
    def execute(self, *_a, **_k):
        raise RuntimeError(f"boom SELECT * FROM invoices -- {SECRET}")  # noqa: S608

    scalar = scalars = execute


@pytest.mark.parametrize(
    ("deps", "code"),
    [
        (_deps_raising(OperationalError("SELECT 1", {}, Exception(SECRET))), "service_unavailable"),
        (
            _deps_raising(DatabaseNotConfiguredError("DATABASE_URL is not set")),
            "service_unavailable",
        ),
        (ToolDependencies(session_scope=lambda: _yield(_ExplodingSession())), "internal_error"),
    ],
    ids=["db_down", "db_not_configured", "unexpected_exception"],
)
def test_failures_are_generic_and_leak_nothing(deps, code, caplog):
    tools = {t.name: t for t in build_commerce_tools(deps)}
    with caplog.at_level(logging.INFO, logger="app.agent.tools"):
        result = invoke_tool(tools["get_invoice"], {"invoice_number": "INV-1001"}, CTX)
    assert result["ok"] is False and result["error"]["code"] == code
    text = json.dumps(result)
    for leak in ("hunter2", "db.internal", "SELECT", "Traceback", "RuntimeError", "Operational"):
        assert leak not in text
    [record] = [r for r in caplog.records if r.getMessage() == "tool call"]
    assert (record.tool, record.error_code) == ("get_invoice", code)
    assert record.tenant_id == str(CTX.tenant_id) and record.request_id == "req-test"
    assert "hunter2" not in caplog.text and "db.internal" not in caplog.text


@contextmanager
def _yield(value):
    yield value
