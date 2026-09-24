"""Action audit events, written in the SAME transaction as the action state change.

Only safe fields: event type, actor (``agent`` / ``system`` / verified user subject), request
id, tool call id, status, decision, failure code, action type, arguments hash.
"""

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.core.request_context import request_id_var
from app.models import ActionRequest, AuditEvent

_ACTOR = {"action_requested": "agent"}


def record_action_event(
    session: Session,
    tenant_id: uuid.UUID,
    event: str,
    row: ActionRequest,
    *,
    decision: str | None = None,
) -> None:
    actor = (
        row.decided_by
        if event == "approval_decided" and row.decided_by
        else _ACTOR.get(event, "system")
    )
    details: dict[str, Any] = {
        "status": str(row.status),
        "action_type": str(row.action_type),
        "arguments_hash": row.arguments_hash,
    }
    if decision:
        details["decision"] = decision
    if row.failure_code:
        details["failure_code"] = row.failure_code
    session.add(
        AuditEvent(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            event_type=event,
            action_request_id=row.id,
            actor=actor[:128],
            request_id=request_id_var.get(),
            tool_call_id=row.tool_call_id,
            details=details,
        )
    )
