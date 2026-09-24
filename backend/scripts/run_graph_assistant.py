"""Developer-only harness: run the LangGraph commerce assistant (Step 6).

    uv run python -m scripts.run_graph_assistant --tenant <uuid> --provider ollama \\
        --text "Show me order ORD-1001"

    # same-process continuation on an EPHEMERAL in-memory thread:
    uv run python -m scripts.run_graph_assistant --tenant <uuid> --thread-id demo \\
        --text "Show me order ORD-1001" --text "Is it paid?"

* --tenant is TRUSTED runtime context (validated before any model is built); it is never
  put into the prompt, graph messages or tool arguments.
* --thread-id enables an InMemorySaver for THIS PROCESS ONLY. The thread and its history
  disappear when the process exits; nothing is written to a database or disk.
* stdout: application-level result JSON (one object, or a list with one per --text).
  stderr: safe log lines / error envelope. Never prints keys, prompts, raw model output,
  reasoning, graph state or raw tool payloads. Gemini: synthetic demo data only.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from langgraph.checkpoint.memory import InMemorySaver

from app.agent.assistant import AssistantError
from app.agent.context import create_agent_context
from app.agent.graph import CommerceGraphAssistant
from app.agent.llm import LLMError, get_llm_provider
from app.agent.llm.config import SUPPORTED_PROVIDERS
from app.core.errors import TenantNotFoundError
from app.db.session import read_only_session
from scripts.run_assistant import _configure_logging, _fail

EPHEMERAL_NOTE = (
    "note: --thread-id uses an in-memory checkpointer; the thread exists only while this "
    "process runs and is lost when it exits."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_graph_assistant", description="Run the LangGraph commerce assistant."
    )
    parser.add_argument("--tenant", type=uuid.UUID, required=True, help="tenant UUID (trusted)")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, help="override LLM_PROVIDER")
    parser.add_argument("--model", help="override the provider's configured model")
    parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="user request (synthetic data only); repeat with --thread-id to continue",
    )
    parser.add_argument("--thread-id", default=None, help="in-memory thread (process lifetime)")
    parser.add_argument("--request-id", default=None, help="optional request id for logs")
    ns = parser.parse_args(argv)
    if len(ns.text) > 1 and ns.thread_id is None:
        parser.error("repeating --text requires --thread-id")
    _configure_logging()

    try:  # trusted context first: unknown tenant fails before any model is built
        with read_only_session() as session:
            context = create_agent_context(session, ns.tenant, ns.request_id)
    except TenantNotFoundError:
        return _fail({"error": {"code": "tenant_not_found", "message": "Unknown tenant."}})
    except ValueError as exc:
        return _fail({"error": {"code": "invalid_request_id", "message": str(exc)}})

    results = []
    try:
        provider = get_llm_provider(provider=ns.provider, model=ns.model)
        checkpointer = InMemorySaver() if ns.thread_id is not None else None
        if checkpointer is not None:
            print(EPHEMERAL_NOTE, file=sys.stderr)
        assistant = CommerceGraphAssistant(provider, checkpointer=checkpointer)
        for turn, text in enumerate(ns.text, start=1):
            try:
                result = assistant.run(text, context, thread_id=ns.thread_id)
            except AssistantError as exc:
                return _fail(
                    {
                        "error": {"code": exc.code, "message": exc.message},
                        "turn": turn,
                        "model_calls": exc.model_calls,
                        "tool_calls": [c.model_dump(mode="json") for c in exc.tool_calls],
                        "invalid_tool_calls": [
                            c.model_dump(mode="json") for c in exc.invalid_tool_calls
                        ],
                    }
                )
            results.append(result.model_dump(mode="json"))
    except LLMError as exc:
        return _fail({"error": {"code": exc.code, "message": exc.message}})
    output = results[0] if len(results) == 1 else results
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
