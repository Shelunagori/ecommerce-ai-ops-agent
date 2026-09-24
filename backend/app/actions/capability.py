"""Model-visible ACTION PROPOSAL capabilities (Step 10).

The model can only PROPOSE; it can never execute. A proposal becomes a persistent
``pending_approval`` request and the graph pauses (LangGraph ``interrupt``) until a
human decides through the approval API. Like ``search_policy_knowledge`` these tools are
schema-only: their functions refuse to run and they are not registered in ``ToolExecutor``.

The model never supplies tenant, idempotency key, status, amounts as floats, or execution
tokens. ``policy_citations`` must be citations retrieved in the CURRENT run; store credit
requires at least one.
"""

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from app.actions.errors import ActionArgumentsInvalidError
from app.models.enums import ActionType

PROPOSE_CANCEL_ORDER = "propose_cancel_order"
PROPOSE_STORE_CREDIT = "propose_store_credit"
ACTION_TOOLS = {
    PROPOSE_CANCEL_ORDER: ActionType.CANCEL_ORDER,
    PROPOSE_STORE_CREDIT: ActionType.ISSUE_STORE_CREDIT,
}
MAX_CITATIONS = 5
_CITATION = re.compile(r"^policy://[^\s]{1,200}$")


class CancelOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_number: str = Field(description="Order reference, e.g. ORD-1004.")
    reason: str = Field(description="Short reason given by the user (max 300 characters).")
    policy_citations: list[str] = Field(
        default_factory=list,
        description="Optional exact policy citations retrieved in this request.",
    )


class StoreCreditInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_code: str = Field(description="Customer reference, e.g. CUS-1002.")
    amount: str = Field(description='Decimal amount as a string, e.g. "15.00".')
    currency: str = Field(description="ISO 4217 currency code, e.g. EUR.")
    reason: str = Field(description="Short reason (max 300 characters).")
    order_number: str | None = Field(default=None, description="Related order, if any.")
    policy_citations: list[str] = Field(
        description="Exact policy citations (retrieved in this request) that justify the credit."
    )


def _refuse(**_kwargs: Any) -> str:
    raise RuntimeError("action proposals are handled only by the graph, never executed as tools")


def action_tools() -> tuple[StructuredTool, ...]:
    return (
        StructuredTool.from_function(
            func=_refuse,
            name=PROPOSE_CANCEL_ORDER,
            description=(
                "Propose cancelling an order. Creates a request that a human must approve; "
                "nothing changes until then. Only orders in draft, confirmed or processing "
                "status can be cancelled."
            ),
            args_schema=CancelOrderInput,
        ),
        StructuredTool.from_function(
            func=_refuse,
            name=PROPOSE_STORE_CREDIT,
            description=(
                "Propose issuing synthetic store credit to a customer. Requires policy "
                "citations retrieved in this request. A human must approve it; nothing "
                "changes until then."
            ),
            args_schema=StoreCreditInput,
        ),
    )


@dataclass(frozen=True)
class ActionProposal:
    action_type: str
    arguments: dict[str, Any]  # business arguments (canonicalised later by the service)
    citations: list[str]


def parse_action_call(name: str, args: Any) -> ActionProposal:
    if name not in ACTION_TOOLS:
        raise ActionArgumentsInvalidError("unknown action capability.")
    if not isinstance(args, dict):
        raise ActionArgumentsInvalidError("arguments must be an object.")
    business = {k: v for k, v in args.items() if k != "policy_citations"}
    raw = args.get("policy_citations", [])
    if raw is None:
        raw = []
    if not isinstance(raw, list) or len(raw) > MAX_CITATIONS:
        raise ActionArgumentsInvalidError(
            f"policy_citations must be a list of at most {MAX_CITATIONS}."
        )
    citations: list[str] = []
    for c in raw:
        if not isinstance(c, str) or not _CITATION.fullmatch(c.strip()):
            raise ActionArgumentsInvalidError("policy_citations must be policy:// citations.")
        if c.strip() not in citations:
            citations.append(c.strip())
    return ActionProposal(ACTION_TOOLS[name].value, business, citations)
