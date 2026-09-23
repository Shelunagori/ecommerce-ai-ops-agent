"""Stable, JSON-serialisable tool result envelope.

Success:  {"ok": true,  "data": <object | list-page | null>}
Failure:  {"ok": false, "error": {"code": "...", "message": "..."}}

Tools return the envelope as a JSON-compatible dict. LangChain serialises it to JSON for
the model's ToolMessage. Only the argument-validation callback (whose LangChain contract is
``str``) returns the same envelope pre-serialised via ``failure_json``. Money is a string,
datetimes are ISO-8601, no ORM objects or exception text ever appear.
"""

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel


class ErrorCode:
    INVALID_ARGUMENTS = "invalid_arguments"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN_TOOL = "unknown_tool"  # set by the assistant executor; the tool never runs
    # Not-found codes come from the service layer: customer_not_found, order_not_found, ...


GENERIC_MESSAGES = {
    ErrorCode.SERVICE_UNAVAILABLE: "The data service is temporarily unavailable.",
    ErrorCode.INTERNAL_ERROR: "The tool failed unexpectedly.",
}


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported tool data type: {type(value).__name__}")


def page(items: Sequence[BaseModel], limit: int) -> dict[str, Any]:
    """Bounded list result. ``items`` may hold limit+1 rows; the extra row sets has_more."""
    visible = list(items)[:limit]
    return {
        "items": [i.model_dump(mode="json") for i in visible],
        "count": len(visible),
        "has_more": len(items) > limit,
    }


Envelope = dict[str, Any]


def success(data: Any) -> Envelope:
    return {"ok": True, "data": data}


def failure(code: str, message: str | None = None) -> Envelope:
    return {"ok": False, "error": {"code": code, "message": message or GENERIC_MESSAGES[code]}}


def failure_json(code: str, message: str | None = None) -> str:
    """The same failure envelope, serialised - only for LangChain callbacks that need ``str``."""
    return json.dumps(failure(code, message), ensure_ascii=False)
