"""Developer-only harness: run the structured intent analysis against a real provider.

    uv run python -m scripts.run_llm --provider ollama --text "Show me order ORD-1001"
    uv run python -m scripts.run_llm --provider gemini --text "Where is SHP-1003?"

* stdout: the result as formatted JSON only. stderr: one safe log line / error.
* Never prints API keys, prompts or raw model responses.
* Gemini: send synthetic/demo text only - never real customer or company data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from app.agent.llm import LLMError, get_llm_provider
from app.agent.llm.config import SUPPORTED_PROVIDERS
from app.agent.llm.intent import analyze_intent
from app.core.logging import JsonFormatter


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "google_genai", "google"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_llm", description="Run structured intent analysis.")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, help="override LLM_PROVIDER")
    parser.add_argument("--model", help="override the provider's configured model")
    parser.add_argument("--text", required=True, help="message to classify (synthetic data only)")
    ns = parser.parse_args(argv)

    _configure_logging()
    try:
        provider = get_llm_provider(provider=ns.provider, model=ns.model)
        result = analyze_intent(provider, ns.text)
    except LLMError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": exc.message}}), file=sys.stderr)
        return 1
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
