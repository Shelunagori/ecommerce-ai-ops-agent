"""Fakes at the provider boundary: a chat model whose structured-output runnable returns
scripted results or raises scripted exceptions. Nothing else is mocked."""

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

from app.agent.llm.provider import ChatModelProvider, ProviderInfo, RetryPolicy
from app.core.config import Settings

FAKE_KEY = "AIzaFAKE-test-key-never-log-9f8e7d"


class FakeChatModel:
    """Duck-types the one BaseChatModel method the provider uses."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def with_structured_output(self, schema: Any, *, method: str, include_raw: bool) -> Any:
        def run(messages: Any) -> Any:
            self.calls.append(
                {
                    "schema": schema,
                    "method": method,
                    "include_raw": include_raw,
                    "messages": messages,
                }
            )
            step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
            if isinstance(step, BaseException):
                raise step
            return step

        return RunnableLambda(run)


def ok(parsed: Any) -> dict[str, Any]:
    return {"raw": object(), "parsed": parsed, "parsing_error": None}


def make_provider(*script: Any, retries: int = 1, sleeps: list[float] | None = None):
    chat = FakeChatModel(*script)
    record = sleeps if sleeps is not None else []
    provider = ChatModelProvider(
        chat,  # type: ignore[arg-type]
        ProviderInfo(provider="fake", model="fake-model"),
        RetryPolicy(max_retries=retries),
        sleep=record.append,
    )
    return provider, chat


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"app_env": "test"}
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Any socket connect / DNS lookup fails the test."""
    attempts: list[str] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        attempts.append(repr(args)[:80])
        raise AssertionError("network I/O attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield attempts
