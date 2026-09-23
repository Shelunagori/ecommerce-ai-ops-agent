"""Ollama models get NO forced reasoning/think setting.

The default is the dedicated non-thinking model qwen3:4b-instruct (Qwen3-4B-Instruct-2507),
so no request-level thinking control is needed. We do not force `think` on any model:
Ollama may reject it for models without thinking support, and it cannot turn a
thinking-only model (e.g. qwen3:4b, which is an alias of qwen3:4b-thinking) into a
non-thinking one.
"""

from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from ollama import Client

from app.agent.llm import get_llm_provider
from tests.llm.conftest import settings


class _CapturedError(Exception):
    pass


@pytest.fixture
def captured_payloads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Intercept the Ollama client's request just before HTTP: records the JSON body."""
    payloads: list[dict[str, Any]] = []

    def fake_request(self, cls, method, path, **kwargs):  # noqa: ANN001
        payloads.append({"path": path, "json": kwargs.get("json")})
        raise _CapturedError

    monkeypatch.setattr(Client, "_request", fake_request)
    return payloads


def _wire_body(chat_model, captured, text="hello"):
    with pytest.raises(Exception):  # noqa: B017 - the interceptor always raises
        chat_model.invoke([HumanMessage(content=text)])
    [req] = captured
    assert req["path"] == "/api/chat"
    return req["json"]


def test_default_is_the_non_thinking_instruct_model(no_network):
    p = get_llm_provider(settings())
    assert (p.info.provider, p.info.model) == ("ollama", "qwen3:4b-instruct")
    assert p._chat_model.reasoning is None  # nothing forced
    assert no_network == []


@pytest.mark.parametrize(
    "model", ["qwen3:4b-instruct", "llama3.2:3b", "qwen3:4b", "qwen3:8b", "mistral:7b"]
)
def test_no_model_gets_a_forced_reasoning_setting(model, no_network):
    p = get_llm_provider(settings(), model=model)
    assert p.info.model == model
    assert p._chat_model.reasoning is None
    assert no_network == []


@pytest.mark.parametrize("model", ["qwen3:4b-instruct", "llama3.2:3b", "qwen3:4b"])
def test_request_body_has_no_think_field(model, captured_payloads, no_network):
    body = _wire_body(get_llm_provider(settings(), model=model)._chat_model, captured_payloads)
    assert body["model"] == model and "think" not in body
    assert no_network == []


def test_default_with_tools_bound_sends_no_think_field(captured_payloads, no_network):
    from app.agent.tools import build_commerce_tools

    bound = get_llm_provider(settings())._chat_model.bind_tools(build_commerce_tools())
    body = _wire_body(bound, captured_payloads, "Show me order ORD-1001")
    assert body["model"] == "qwen3:4b-instruct" and len(body["tools"]) == 12
    assert "think" not in body
    assert no_network == []


def test_no_model_specific_reasoning_rule_remains_in_the_factory():
    import app.agent.llm.factory as factory

    assert not hasattr(factory, "ollama_reasoning_kwargs")
    assert not hasattr(factory, "_OLLAMA_NON_THINKING_FAMILIES")
