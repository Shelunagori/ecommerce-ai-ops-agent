"""OPT-IN live LangGraph smoke test (compatibility, not model quality).

    RUN_OLLAMA_INTEGRATION=1 (default model qwen3:4b-instruct) / RUN_GEMINI_INTEGRATION=1
    plus TEST_DATABASE_URL.

Checks the real model requests a PARSED tool call, the real tool runs against the synthetic
test database, the graph loops back to the model and ends with a non-empty answer that
exposes no reasoning, protocol artifact or tenant/runtime data. Exact prose is not asserted.
"""

import os

import pytest
from sqlalchemy.orm import sessionmaker

import app.db.session as db_session_module
from app.agent.assistant import AssistantError, AssistantLimits
from app.agent.assistant.answer import detect_protocol_artifact
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from app.agent.llm import get_llm_provider
from app.agent.tools import COMMERCE_TOOL_NAMES, ToolDependencies, build_commerce_tools

PROVIDERS = [
    pytest.param(
        "ollama",
        marks=pytest.mark.skipif(
            os.getenv("RUN_OLLAMA_INTEGRATION") != "1", reason="set RUN_OLLAMA_INTEGRATION=1"
        ),
    ),
    pytest.param(
        "gemini",
        marks=pytest.mark.skipif(
            os.getenv("RUN_GEMINI_INTEGRATION") != "1", reason="set RUN_GEMINI_INTEGRATION=1"
        ),
    ),
]


@pytest.fixture
def live_tools(db_engine, monkeypatch):
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_session_module, "_session_factory", lambda: factory)
    return build_commerce_tools(ToolDependencies(session_scope=db_session_module.read_only_session))


@pytest.mark.llm_integration
@pytest.mark.parametrize("provider_name", PROVIDERS)
def test_live_graph_order_lookup(provider_name, live_tools, tenant_a):
    ctx = AgentContext(tenant_a.tenant_id, "live-graph")
    assistant = CommerceGraphAssistant(
        get_llm_provider(provider=provider_name), tools=live_tools, limits=AssistantLimits()
    )
    assert assistant.bound_tool_names == COMMERCE_TOOL_NAMES
    try:
        result = assistant.run("Show me order ORD-1001", ctx)
    except AssistantError as exc:
        pytest.fail(f"graph run did not finish: {exc.code} ({exc.detail})")
    assert result.provider == provider_name and 2 <= result.model_calls <= 5  # looped to MODEL
    assert "get_order" in [c.tool for c in result.tool_calls]
    assert any(c.tool == "get_order" and c.outcome == "success" for c in result.tool_calls)
    answer = result.answer
    assert answer.strip() and detect_protocol_artifact(answer) is None
    for leak in (
        "<think>",
        "</think>",
        str(tenant_a.tenant_id),
        "tenant_id",
        "ToolRuntime",
        "live-graph",
    ):
        assert leak not in answer
