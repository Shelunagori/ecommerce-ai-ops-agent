"""Developer-only harness: invoke one registered agent tool without a model.

    uv run python -m scripts.run_tool --tenant <uuid> --tool get_order \\
        --args '{"order_number": "ORD-1001"}'
    uv run python -m scripts.run_tool --list

* Only tools from the registry can be run; arguments are validated by the tool's schema.
* The tenant comes from --tenant (trusted CLI context), never from --args; the tenant
  must exist. A ``tenant_id``/``runtime`` key inside --args is rejected by the schema.
* Prints the tool's JSON envelope to stdout; structured logs go to stderr.
* Development/demo tooling only - not an HTTP endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid

from app.agent.context import create_agent_context
from app.agent.tools import COMMERCE_TOOL_NAMES, build_commerce_tools
from app.agent.tools.invoke import invoke_tool
from app.core.errors import TenantNotFoundError
from app.core.logging import JsonFormatter, RequestContextFilter
from app.db.session import read_only_session


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_tool", description="Invoke one read-only agent tool.")
    p.add_argument("--list", action="store_true", help="list tools and their model-visible schemas")
    p.add_argument("--tenant", type=uuid.UUID, help="tenant UUID (trusted runtime context)")
    p.add_argument("--tool", choices=COMMERCE_TOOL_NAMES, help="registered tool name")
    p.add_argument("--args", default="{}", help="JSON object of business arguments")
    p.add_argument("--request-id", default=None, help="optional request id for log correlation")
    return p


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestContextFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    ns = parser.parse_args(argv)
    tools = {t.name: t for t in build_commerce_tools()}

    if ns.list:
        for tool in tools.values():
            schema = tool.tool_call_schema.model_json_schema()
            print(
                json.dumps(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema.get("properties", {}),
                    }
                )
            )
        return 0
    if ns.tenant is None or ns.tool is None:
        parser.error("--tenant and --tool are required (or use --list)")

    try:
        args = json.loads(ns.args)
    except json.JSONDecodeError:
        parser.error("--args must be valid JSON")
    if not isinstance(args, dict):
        parser.error("--args must be a JSON object")

    _configure_logging()
    try:
        with read_only_session() as session:
            context = create_agent_context(session, ns.tenant, ns.request_id)
    except TenantNotFoundError:
        print("Unknown tenant.", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    result = invoke_tool(tools[ns.tool], args, context)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
