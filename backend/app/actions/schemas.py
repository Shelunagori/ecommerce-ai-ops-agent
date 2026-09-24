"""Model-supplied action arguments -> strict, canonical, hashable arguments.

The model proposes; this module validates. Canonical arguments contain JSON primitives only
(money as a normalised ``"12.34"`` string, never a float). ``arguments_hash`` is the sha256
of the canonical JSON of {action_type, arguments, evidence citations}: what a human
approves is bound to exactly what later executes.
"""

import hashlib
import json
import re
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Any

from app.actions.errors import ActionArgumentsInvalidError
from app.models.enums import ActionType

_REF = re.compile(r"^[A-Z0-9][A-Z0-9-]{0,31}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_AMOUNT = re.compile(r"^\d{1,7}(\.\d{1,2})?$")
MAX_REASON_CHARS = 300
CENT = Decimal("0.01")


def _reference(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _REF.fullmatch(value.strip().upper()):
        raise ActionArgumentsInvalidError(f"{field} must be a reference like ORD-1004.")
    return value.strip().upper()


def _reason(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActionArgumentsInvalidError("reason must be a non-empty string.")
    text = " ".join(value.split())  # collapse whitespace/newlines (display-safe)
    if len(text) > MAX_REASON_CHARS:
        raise ActionArgumentsInvalidError(f"reason must be at most {MAX_REASON_CHARS} characters.")
    return text


def parse_amount(value: Any) -> Decimal:
    """Decimal only: accepts a decimal STRING ("15" / "15.5" / "15.00"). Floats and
    ints are rejected so no binary floating point ever reaches money."""
    if not isinstance(value, str) or not _AMOUNT.fullmatch(value.strip()):
        raise ActionArgumentsInvalidError('amount must be a decimal string such as "15.00".')
    try:
        amount = Decimal(value.strip()).quantize(CENT, rounding=ROUND_HALF_EVEN)
    except InvalidOperation:
        raise ActionArgumentsInvalidError("amount is not a valid decimal.") from None
    if amount <= 0:
        raise ActionArgumentsInvalidError("amount must be positive.")
    return amount


def _only(args: Any, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ActionArgumentsInvalidError("arguments must be an object.")
    extra = sorted(k if isinstance(k, str) else "<invalid>" for k in args if k not in allowed)
    if extra:
        raise ActionArgumentsInvalidError(f"unexpected argument(s): {', '.join(extra[:10])}.")
    missing = sorted(required - set(args))
    if missing:
        raise ActionArgumentsInvalidError(f"missing argument(s): {', '.join(missing)}.")
    return args


def canonical_cancel_order(args: Any) -> dict[str, Any]:
    a = _only(args, {"order_number", "reason"}, {"order_number", "reason"})
    return {
        "order_number": _reference(a["order_number"], "order_number"),
        "reason": _reason(a["reason"]),
    }


def canonical_store_credit(args: Any) -> dict[str, Any]:
    a = _only(
        args,
        {"customer_code", "amount", "currency", "reason", "order_number"},
        {"customer_code", "amount", "currency", "reason"},
    )
    currency = a["currency"].strip().upper() if isinstance(a["currency"], str) else None
    if currency is None or not _CURRENCY.fullmatch(currency):
        raise ActionArgumentsInvalidError("currency must be an ISO 4217 code such as EUR.")
    order = a.get("order_number")
    return {
        "customer_code": _reference(a["customer_code"], "customer_code"),
        "amount": str(parse_amount(a["amount"])),
        "currency": currency,
        "reason": _reason(a["reason"]),
        "order_number": None if order in (None, "") else _reference(order, "order_number"),
    }


CANONICALIZERS = {
    ActionType.CANCEL_ORDER: canonical_cancel_order,
    ActionType.ISSUE_STORE_CREDIT: canonical_store_credit,
}


def canonical_arguments(action_type: str, args: Any) -> dict[str, Any]:
    try:
        canonicalize = CANONICALIZERS[ActionType(action_type)]
    except (ValueError, KeyError):
        raise ActionArgumentsInvalidError("unknown action type.") from None
    return canonicalize(args)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def arguments_hash(
    action_type: str, arguments: dict[str, Any], evidence: list[dict[str, Any]]
) -> str:
    payload = {
        "action_type": action_type,
        "arguments": arguments,
        "evidence": sorted(e["citation"] for e in evidence),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def derive_idempotency_key(tenant_id: Any, thread_key: str, tool_call_id: str) -> str:
    """Stable per proposal: the same tool call on the same thread maps to the same key, so
    a re-run proposal node can never create a second request."""
    raw = f"{tenant_id}:{thread_key}:{tool_call_id}"
    return "idk-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:64]
