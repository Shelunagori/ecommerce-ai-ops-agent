"""The production agent runtime behind the API (Phase 6).

One per application process, built lazily on first use (no network or DB at import/startup):

* chat model from settings (``get_llm_provider``; hosted in production),
* ``AGENT_PROFILE`` graph (prompt v3, commerce tools, policy RAG, approval-gated actions),
* DURABLE PostgreSQL checkpoints (``PostgresSaver``) so approvals survive restarts and can be
  resumed by any API instance,
* ``ActionService`` on read-write units of work, durable run records (``RunRecorder``),
* a second, READ-ONLY assistant (``PUBLIC_DEMO_PROFILE``: no action tools) for verified
  anonymous public-demo visitors, sharing the same checkpointer.

Tests replace ``app.state.agent_runtime`` with an in-memory runtime (scripted model).
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any

from app.actions.service import ActionService
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.profile import AGENT_PROFILE, PUBLIC_DEMO_PROFILE
from app.auth.errors import PublicDemoUnavailableError


@dataclass
class AgentRuntime:
    assistant: CommerceGraphAssistant
    actions: ActionService
    closer: Any = None
    public_demo: CommerceGraphAssistant | None = None  # read-only profile (no action tools)

    def assistant_for(self, principal: Any) -> CommerceGraphAssistant:
        """The capability set is chosen by the VERIFIED principal, never by the request."""
        if getattr(principal, "public_demo", False):
            if self.public_demo is None:
                raise PublicDemoUnavailableError()
            return self.public_demo
        return self.assistant

    def close(self) -> None:
        if self.closer is not None:
            self.closer()


_lock = threading.Lock()


def build_agent_runtime(settings: Any) -> AgentRuntime:
    from app.agent.graph.checkpoint import durable_checkpointer  # noqa: PLC0415
    from app.agent.llm import get_llm_provider  # noqa: PLC0415
    from app.observability.runs import RunRecorder  # noqa: PLC0415

    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required for the agent runtime")
    checkpointer = durable_checkpointer(settings.database_url)
    actions = ActionService()
    provider = get_llm_provider(settings)
    assistant = CommerceGraphAssistant(
        provider,
        checkpointer=checkpointer.saver,
        profile=AGENT_PROFILE,
        actions=actions,
        run_recorder=RunRecorder(),
    )
    public_demo = (
        CommerceGraphAssistant(
            provider,
            checkpointer=checkpointer.saver,
            profile=PUBLIC_DEMO_PROFILE,
            run_recorder=RunRecorder(),
        )
        if settings.public_demo_enabled
        else None
    )
    return AgentRuntime(assistant, actions, closer=checkpointer.close, public_demo=public_demo)


def runtime_for(app: Any) -> AgentRuntime:
    runtime = getattr(app.state, "agent_runtime", None)
    if runtime is None:
        with _lock:
            runtime = getattr(app.state, "agent_runtime", None)
            if runtime is None:
                runtime = build_agent_runtime(app.state.settings)
                app.state.agent_runtime = runtime
    return runtime


def internal_thread_id(subject: str, thread_id: str) -> str:
    """Per-USER thread namespace inside the tenant: two users of one tenant never share a
    thread, even with the same client thread id. (The tenant is added by the runner's
    ``cg1-sha256(tenant:thread)`` key.)"""
    tag = hashlib.sha256(subject.encode("utf-8")).hexdigest()[:12]
    return f"u{tag}-{thread_id}"
