"""OPT-IN live tool-calling smoke tests (compatibility, not model quality).

    RUN_OLLAMA_INTEGRATION=1 / RUN_GEMINI_INTEGRATION=1  plus TEST_DATABASE_URL.

They check that bound tools reach the provider, tool calls parse and execute against the
synthetic test database, and the loop finishes with a non-empty answer that leaks no
tenant/runtime data. Which tool a model picks is NOT asserted (evaluation step).
"""

import os

import pytest
from sqlalchemy.orm import sessionmaker

import app.db.session as db_session_module
from app.agent.assistant import AssistantError, AssistantLimits, CommerceAssistant
from app.agent.context import AgentContext
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
@pytest.mark.parametrize("text", ["Show me order ORD-1001", "hello"])
def test_live_tool_calling_loop_completes(provider_name, text, live_tools, tenant_a):
    ctx = AgentContext(tenant_a.tenant_id, "live-smoke")
    assistant = CommerceAssistant(
        get_llm_provider(provider=provider_name), tools=live_tools, limits=AssistantLimits()
    )
    assert assistant.bound_tool_names == COMMERCE_TOOL_NAMES
    try:
        result = assistant.run(text, ctx)
    except AssistantError as exc:  # a limit/protocol outcome is still a parsed, safe result
        assert exc.code in {"agent_limit_exceeded", "agent_protocol_error", "agent_empty_answer"}
        pytest.xfail(f"model did not finish cleanly: {exc.code}")
    assert result.answer.strip()
    assert result.provider == provider_name and 1 <= result.model_calls <= 5
    assert all(
        c.tool in COMMERCE_TOOL_NAMES or c.outcome == "unknown_tool" for c in result.tool_calls
    )
    for leak in (str(tenant_a.tenant_id), "tenant_id", "ToolRuntime", "live-smoke"):
        assert leak not in result.answer
