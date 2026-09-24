"""``search_policy_knowledge``: the one model-visible knowledge capability (Step 9).

Model-controlled arguments (untrusted, validated here):

* ``query``  - 1-500 characters of search text;
* ``as_of``  - optional ISO calendar date ``YYYY-MM-DD`` taken from the user's request
  (e.g. "on 10 June 2026"). It is information, not authorization: it only selects which
  policy VERSION applies; tenant and effective-date filtering stay server-side.

Never model-controlled: tenant, embedding provider/profile, result limit (fixed:
``POLICY_RETRIEVAL_LIMIT``), database filters, citations, file paths. Any extra argument
(``tenant_id``, ``limit``, ...) makes the call invalid - nothing is executed.

The ``StructuredTool`` returned by ``policy_search_tool()`` exists only so providers can
bind its schema. Its function refuses to run: execution happens in the graph's RETRIEVE
node, never through LangChain tool invocation or the commerce ``ToolExecutor``.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

SEARCH_POLICY_KNOWLEDGE = "search_policy_knowledge"
POLICY_RETRIEVAL_LIMIT = 3  # trusted, fixed for Step 9
MAX_POLICY_QUERY_CHARS = 500
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_]{1,64}$")
ALLOWED_ARGUMENTS = frozenset({"query", "as_of"})

DESCRIPTION = (
    "Search this company's policy knowledge (compensation, refunds, returns, cancellation, "
    "shipping rules, eligibility). Returns the applicable policy sections with citations. "
    "Use for policy questions, not for customer/order/invoice/shipment/product facts."
)


class PolicySearchInput(BaseModel):
    """Schema shown to the model (and nothing else)."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        description="Short search text describing the policy question, max 500 characters."
    )
    as_of: str | None = Field(
        default=None,
        description=(
            "Optional ISO date YYYY-MM-DD when the user asks about a specific past or future "
            "date. Omit for the current policy."
        ),
    )


def _refuse(**_kwargs: Any) -> str:
    raise RuntimeError(
        "search_policy_knowledge is executed only by the graph RETRIEVE node, never as a tool"
    )


def policy_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_refuse,
        name=SEARCH_POLICY_KNOWLEDGE,
        description=DESCRIPTION,
        args_schema=PolicySearchInput,
    )


@dataclass(frozen=True)
class PolicySearchArgs:
    query: str
    as_of: date | None


class PolicySearchArgumentError(ValueError):
    """Model-supplied arguments are invalid. ``str()`` is a short, safe, model-facing reason."""

    def __init__(self, reason: str, rejected: list[str] | None = None) -> None:
        self.reason = reason
        self.rejected = sorted(rejected or [])
        super().__init__(reason)


def parse_policy_search_args(args: Any) -> PolicySearchArgs:
    if not isinstance(args, dict):
        raise PolicySearchArgumentError("arguments must be an object")
    extra = [k for k in args if k not in ALLOWED_ARGUMENTS]
    if extra:
        names = [k if isinstance(k, str) and _SAFE_KEY.fullmatch(k) else "<invalid>" for k in extra]
        raise PolicySearchArgumentError(
            "unexpected argument(s): " + ", ".join(sorted(names)[:10]), names[:10]
        )
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise PolicySearchArgumentError("query must be a non-empty string")
    query = query.strip()
    if len(query) > MAX_POLICY_QUERY_CHARS:
        raise PolicySearchArgumentError(
            f"query must be at most {MAX_POLICY_QUERY_CHARS} characters"
        )
    raw = args.get("as_of")
    if raw is None:
        return PolicySearchArgs(query, None)
    if not isinstance(raw, str) or not _ISO_DATE.fullmatch(raw):
        raise PolicySearchArgumentError("as_of must be an ISO date YYYY-MM-DD")
    try:
        as_of = date.fromisoformat(raw)
    except ValueError:
        raise PolicySearchArgumentError("as_of must be a valid calendar date") from None
    return PolicySearchArgs(query, as_of)
