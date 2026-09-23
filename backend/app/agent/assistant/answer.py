"""Final-answer guard: reject tool-protocol artifacts that arrive as plain text.

Tools run ONLY from LangChain's parsed ``AIMessage.tool_calls``. When a model instead writes
a pseudo tool call into its text (a JSON ``{"name": ..., "parameters": ...}`` object or
``<tool_call>`` markup), or returns a bare ``{}`` / ``[]``, that text is not a
user-facing answer. It is classified here - never parsed for execution - and the run
ends with ``agent_protocol_error``.

Deliberately narrow: only these shapes are recognised; other text (including ordinary JSON
that does not look like a call) is left alone.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from app.agent.assistant.executor import safe_name

ArtifactKind = Literal["empty_structured_output", "textual_tool_call"]

_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?```$", re.S)
_TOOL_MARKUP = re.compile(
    r"<\s*(tool_call|function_call)\b[^>]*>(.*?)(<\s*/\s*\1\s*>|$)", re.S | re.I
)
_ARGUMENT_KEYS = ("parameters", "arguments", "args")


@dataclass(frozen=True)
class ProtocolArtifact:
    kind: ArtifactKind
    name: str | None  # sanitised attempted tool name, for evaluation only


def detect_protocol_artifact(text: str) -> ProtocolArtifact | None:
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    body = fenced.group(1).strip() if fenced else stripped

    markup = _TOOL_MARKUP.search(body)
    if markup:
        return ProtocolArtifact("textual_tool_call", _call_name(_loads(markup.group(2).strip())))

    value = _loads(body)
    if value is _NOT_JSON:
        return None
    if value in ({}, []):
        return ProtocolArtifact("empty_structured_output", None)
    candidates = value if isinstance(value, list) else [value]
    if candidates and all(_looks_like_call(c) for c in candidates):
        return ProtocolArtifact("textual_tool_call", _call_name(candidates[0]))
    return None


_NOT_JSON = object()


def _loads(text: str) -> Any:
    if not text or text[0] not in "{[":
        return _NOT_JSON
    try:
        return json.loads(text)
    except ValueError:
        return _NOT_JSON


def _looks_like_call(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("name"), str)
        and any(k in value for k in _ARGUMENT_KEYS)
    )


def _call_name(value: Any) -> str | None:
    if isinstance(value, list) and value:
        value = value[0]
    return safe_name(value.get("name")) if isinstance(value, dict) else None
