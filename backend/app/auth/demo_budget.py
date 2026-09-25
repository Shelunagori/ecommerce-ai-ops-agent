"""Durable per-visitor message budget for the public demo (anonymous visitors only).

``reserve`` atomically claims one message (a single conditional UPDATE: concurrent requests
of the same visitor cannot overshoot the limit), ``release`` gives it back when the run
fails, so the budget counts messages that actually ran. Enforced server-side from the
VERIFIED JWT subject; nothing the browser sends is trusted. PostgreSQL only - no Redis.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager

from sqlalchemy import func, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import PublicDemoUsage

logger = logging.getLogger("app.auth")


def subject_digest(subject: str) -> str:
    return hashlib.sha256(f"public-demo:{subject}".encode()).hexdigest()


class PublicDemoBudget:
    def __init__(
        self, session_scope: Callable[[], AbstractContextManager[Session]] | None = None
    ) -> None:
        if session_scope is None:
            from app.db.session import unit_of_work  # noqa: PLC0415

            session_scope = unit_of_work
        self._scope = session_scope

    def reserve(self, subject: str, limit: int) -> bool:
        """Claim one message; False when the visitor's budget is used up. ``limit`` 0 = off."""
        if limit <= 0:
            return True
        key = subject_digest(subject)
        table = PublicDemoUsage.__table__
        with self._scope() as session:
            session.execute(
                insert(table).values(subject_hash=key, messages=0).on_conflict_do_nothing()
            )
            claimed = session.execute(
                update(table)
                .where(table.c.subject_hash == key, table.c.messages < limit)
                .values(messages=table.c.messages + 1, last_used_at=func.now())
                .returning(table.c.messages)
            ).first()
        return claimed is not None

    def release(self, subject: str) -> None:
        """Give one message back (the run failed). Best effort: never fails the request."""
        table = PublicDemoUsage.__table__
        try:
            with self._scope() as session:
                session.execute(
                    update(table)
                    .where(table.c.subject_hash == subject_digest(subject), table.c.messages > 0)
                    .values(messages=table.c.messages - 1)
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "public demo budget not released", extra={"error_type": type(exc).__name__}
            )

    def used(self, subject: str) -> int:
        with self._scope() as session:
            row = session.get(PublicDemoUsage, subject_digest(subject))
            return row.messages if row else 0
