"""Deterministic fake chat model for the assistant loop (only the MODEL is fake).

It plays a script of steps. Each step is an AIMessage, an exception, or a callable
``(messages) -> AIMessage`` so a step can react to the ToolMessages it was given.
Every invocation records a snapshot of what the model was sent and which tools were bound.
"""

import itertools
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from sqlalchemy.exc import OperationalError

from app.agent.llm.provider import ChatModelProvider, ProviderInfo, RetryPolicy
from app.agent.tools import ToolDependencies, build_commerce_tools

Step = AIMessage | BaseException | Callable[[list[BaseMessage]], AIMessage]

_ids = itertools.count(1)


def call(name: str, call_id: str | None = None, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id or f"call_{next(_ids)}", "type": "tool_call"}


def ai_tools(*calls: dict[str, Any], **extra: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls), **extra)


def ai_text(text: str) -> AIMessage:
    return AIMessage(content=text)


@dataclass
class Invocation:
    messages: list[BaseMessage]
    tool_names: tuple[str, ...]


@dataclass
class ScriptedChatModel:
    script: list[Step]
    invocations: list[Invocation] = field(default_factory=list)
    bound: tuple[str, ...] = ()

    def bind_tools(self, tools: Sequence[BaseTool]) -> "ScriptedChatModel":
        self.bound = tuple(t.name for t in tools)
        return self

    def invoke(self, messages: list[BaseMessage]) -> Any:
        self.invocations.append(Invocation(list(messages), self.bound))
        if not self.script:
            raise AssertionError("fake model script exhausted")
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step(list(messages)) if callable(step) and not isinstance(step, AIMessage) else step


def make_provider(*script: Step, retries: int = 1) -> tuple[ChatModelProvider, ScriptedChatModel]:
    model = ScriptedChatModel(list(script))
    provider = ChatModelProvider(
        model,  # type: ignore[arg-type]
        ProviderInfo(provider="fake", model="scripted"),
        RetryPolicy(max_retries=retries),
        sleep=lambda _s: None,
    )
    return provider, model


class DownDatabase:
    """Real registry, but every DB session fails -> deterministic service_unavailable."""

    def __init__(self) -> None:
        self.sessions_opened = 0

    @contextmanager
    def scope(self):
        self.sessions_opened += 1
        raise OperationalError("SELECT 1", {}, Exception("db down"))
        yield  # pragma: no cover


def offline_tools(db: DownDatabase) -> list[BaseTool]:
    return build_commerce_tools(ToolDependencies(session_scope=db.scope))
