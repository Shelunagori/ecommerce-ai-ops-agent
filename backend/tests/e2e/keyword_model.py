"""Deterministic keyword-driven chat model for the local end-to-end harness.

Only the MODEL is fake. It reads the latest user message and the ToolMessages the graph has
fed back in this run, and emits the tool calls / answers a well-behaved model would. Every
other layer (routing, tools, retrieval, grounding, approvals, persistence) is real.

Recognised requests (case-insensitive):
* ``cancel ORD-nnnn``                  -> propose_cancel_order (approval-gated)
* ``credit CUS-nnnn [for ORD-nnnn]``   -> policy search, then propose_store_credit citing it
* ``SHP-nnnn`` + a policy word        -> get_shipment, THEN policy search, cited answer
  (commerce first: after retrieval no new commerce tool may run in the turn)
* ``polic`` / ``compensation`` / ``refund`` / ``return`` -> policy search, cited answer
* ``SHP-nnnn``                         -> get_shipment, answer with its status
* ``ORD-nnnn``                         -> get_order, answer with its status
* anything else                        -> a short greeting
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.actions.capability import PROPOSE_CANCEL_ORDER, PROPOSE_STORE_CREDIT
from app.agent.rag.capability import SEARCH_POLICY_KNOWLEDGE

GREETING = (
    "Hello! I can look up orders, answer policy questions with citations, and propose "
    "order cancellations or store credit for human approval."
)
_ORDER = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)
_CUSTOMER = re.compile(r"\bCUS-\d{4}\b", re.IGNORECASE)
_SHIPMENT = re.compile(r"\bSHP-\d{4}\b", re.IGNORECASE)
_POLICY_WORDS = ("polic", "compensation", "refund", "return")
_ids = itertools.count(1)


def _call(name: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"e2e_{next(_ids)}", "type": "tool_call"}],
    )


def _citations(message: ToolMessage) -> list[str]:
    artifact = message.artifact if isinstance(message.artifact, dict) else {}
    return list(artifact.get("policy_citations") or [])


@dataclass
class KeywordChatModel:
    bound: tuple[str, ...] = ()

    def bind_tools(self, tools: Sequence[BaseTool]) -> KeywordChatModel:
        self.bound = tuple(t.name for t in tools)
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        last = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        text = str(messages[last].content)
        lowered = text.lower()
        results = [m for m in messages[last + 1 :] if isinstance(m, ToolMessage)]
        order = _ORDER.search(text)
        customer = _CUSTOMER.search(text)

        if "cancel" in lowered and order:
            if PROPOSE_CANCEL_ORDER not in self.bound:  # read-only (public demo) profile
                return AIMessage(
                    content="The public demo is read-only, so I cannot cancel orders. "
                    "A reviewer account with approval rights can propose this action."
                )
            if not results:
                return _call(
                    PROPOSE_CANCEL_ORDER,
                    order_number=order.group(0).upper(),
                    reason="Customer requested cancellation",
                )
            return AIMessage(content="I could not propose that cancellation.")

        if "credit" in lowered and customer:
            if not results:
                return _call(SEARCH_POLICY_KNOWLEDGE, query="delayed shipment store credit")
            cited = _citations(results[0])
            if len(results) == 1 and cited:
                args: dict[str, Any] = {
                    "customer_code": customer.group(0).upper(),
                    "amount": "5.00",
                    "currency": "USD",
                    "reason": "Delayed shipment compensation",
                    "policy_citations": cited[:1],
                }
                if order:
                    args["order_number"] = order.group(0).upper()
                return _call(PROPOSE_STORE_CREDIT, **args)
            return AIMessage(content="I could not propose that store credit.")

        shipment = _SHIPMENT.search(text)
        if shipment and any(k in lowered for k in _POLICY_WORDS):
            number = shipment.group(0).upper()
            if not results:
                return _call("get_shipment", shipment_number=number)
            if len(results) == 1:
                return _call(SEARCH_POLICY_KNOWLEDGE, query="delayed shipment compensation")
            try:
                status = (json.loads(str(results[0].content)).get("data") or {}).get("status")
            except ValueError:
                status = None
            cited = _citations(results[1])
            fact = f"Shipment {number} is {status}." if status else f"I could not find {number}."
            if not cited:
                return AIMessage(content=f"{fact} I could not find a policy that covers this.")
            return AIMessage(content=f"{fact} The compensation policy applies [{cited[0]}].")

        if shipment:  # shipment only (no policy question)
            number = shipment.group(0).upper()
            if not results:
                return _call("get_shipment", shipment_number=number)
            try:
                status = (json.loads(str(results[0].content)).get("data") or {}).get("status")
            except ValueError:
                status = None
            return AIMessage(
                content=f"Shipment {number} is {status}."
                if status
                else f"I could not find {number}."
            )

        if any(k in lowered for k in _POLICY_WORDS):
            if not results:
                return _call(SEARCH_POLICY_KNOWLEDGE, query=text)
            cited = _citations(results[0])
            if not cited:
                return AIMessage(content="I could not find a policy that covers this.")
            return AIMessage(content=f"Here is what the current policy says [{cited[0]}].")

        if order:
            if not results:
                return _call("get_order", order_number=order.group(0).upper())
            try:
                payload = json.loads(str(results[0].content))
            except ValueError:
                payload = {}
            status = (payload.get("data") or {}).get("status")
            if status:
                return AIMessage(content=f"Order {order.group(0).upper()} is {status}.")
            return AIMessage(content=f"I could not find order {order.group(0).upper()}.")

        return AIMessage(content=GREETING)
