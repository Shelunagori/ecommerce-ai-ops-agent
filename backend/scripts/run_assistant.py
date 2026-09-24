"""Developer-only harness: run the model-driven commerce assistant once.

    uv run python -m scripts.run_assistant --tenant <uuid> --provider ollama \\
        --text "Show me order ORD-1001"

* --tenant is TRUSTED runtime context (validated before the model is built); it is never
  put into the prompt or into tool arguments.
* stdout: result JSON (answer, provider/model, prompt version, model calls, tool-call
  summary, duration). stderr: safe log lines / error envelope.
* Never prints keys, prompts, raw model output, reasoning or raw tool payloads.
* Gemini: synthetic demo data only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid

from app.agent.assistant import AssistantError, CommerceAssistant
from app.agent.context import create_agent_context
from app.agent.llm import LLMError, get_llm_provider
from app.agent.llm.config import SUPPORTED_PROVIDERS
from app.core.errors import TenantNotFoundError
from app.core.logging import JsonFormatter, RequestContextFilter
from app.db.session import read_only_session


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestContextFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "google_genai", "google"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _fail(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_assistant", description="Run the commerce assistant."
    )
    parser.add_argument("--tenant", type=uuid.UUID, required=True, help="tenant UUID (trusted)")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, help="override LLM_PROVIDER")
    parser.add_argument("--model", help="override the provider's configured model")
    parser.add_argument("--text", required=True, help="user request (synthetic data only)")
    parser.add_argument("--request-id", default=None, help="optional request id for logs")
    ns = parser.parse_args(argv)
    _configure_logging()

    try:  # trusted context first: unknown tenant fails before any model is built
        with read_only_session() as session:
            context = create_agent_context(session, ns.tenant, ns.request_id)
    except TenantNotFoundError:
        return _fail({"error": {"code": "tenant_not_found", "message": "Unknown tenant."}})
    except ValueError as exc:
        return _fail({"error": {"code": "invalid_request_id", "message": str(exc)}})

    try:
        provider = get_llm_provider(provider=ns.provider, model=ns.model)
        result = CommerceAssistant(provider).run(ns.text, context)
    except LLMError as exc:
        return _fail({"error": {"code": exc.code, "message": exc.message}})
    except AssistantError as exc:
        return _fail(
            {
                "error": {"code": exc.code, "message": exc.message},
                "model_calls": exc.model_calls,
                "tool_calls": [c.model_dump(mode="json") for c in exc.tool_calls],
                "invalid_tool_calls": [c.model_dump(mode="json") for c in exc.invalid_tool_calls],
            }
        )
    # Step-5 output shape is unchanged: the RAG-only fields (always empty here) are omitted.
    shown = result.model_dump(mode="json", exclude={"retrievals", "citations", "action"})
    print(json.dumps(shown, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
