"""Best-effort durable run records (``agent_runs``).

A failure to record never fails the user's run (it is logged as ``agent run not recorded``);
action audit events, in contrast, are transactional with the action itself.
"""

import logging
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from sqlalchemy.orm import Session

from app.models import AgentRun

logger = logging.getLogger("app.observability")

_FIELDS = (
    "kind",
    "request_id",
    "thread_key",
    "runner",
    "profile",
    "prompt_version",
    "provider",
    "model",
    "outcome",
    "error_detail",
    "model_calls",
    "commerce_tool_count",
    "policy_retrieval_count",
    "retrieved_citation_count",
    "final_citation_count",
    "grounding_failure",
    "action_request_id",
    "action_status",
    "duration_ms",
)


class RunRecorder:
    def __init__(self, session_scope: Callable[[], AbstractContextManager[Session]] | None = None):
        if session_scope is None:
            from app.db.session import unit_of_work  # noqa: PLC0415

            session_scope = unit_of_work
        self._scope = session_scope

    def __call__(self, tenant_id: uuid.UUID, record: dict[str, Any]) -> None:
        try:
            with self._scope() as session:
                session.add(
                    AgentRun(
                        id=uuid.uuid4(), tenant_id=tenant_id, **{k: record.get(k) for k in _FIELDS}
                    )
                )
        except Exception as exc:  # noqa: BLE001 - observability must not break the run
            logger.warning("agent run not recorded", extra={"error_type": type(exc).__name__})
