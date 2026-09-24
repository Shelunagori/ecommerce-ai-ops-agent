"""The model-visible policy capability: schema, argument parsing, isolation from ToolExecutor."""

from datetime import date

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agent.assistant.executor import ToolExecutor
from app.agent.rag.capability import (
    MAX_POLICY_QUERY_CHARS,
    POLICY_RETRIEVAL_LIMIT,
    SEARCH_POLICY_KNOWLEDGE,
    PolicySearchArgumentError,
    parse_policy_search_args,
    policy_search_tool,
)
from app.agent.tools import COMMERCE_TOOL_NAMES, build_commerce_tools


def test_model_visible_schema_has_only_query_and_as_of():
    schema = convert_to_openai_tool(policy_search_tool())["function"]
    assert schema["name"] == SEARCH_POLICY_KNOWLEDGE
    props = schema["parameters"]["properties"]
    assert set(props) == {"query", "as_of"}
    assert schema["parameters"]["required"] == ["query"]
    flat = str(schema).lower()
    for forbidden in ("tenant", "limit", "profile", "provider", "citation", "path", "filter"):
        assert forbidden not in set(props) and f"'{forbidden}'" not in flat


def test_fixed_trusted_limit():
    assert POLICY_RETRIEVAL_LIMIT == 3 and MAX_POLICY_QUERY_CHARS == 500


def test_schema_only_tool_cannot_be_executed():
    with pytest.raises(RuntimeError):
        policy_search_tool().invoke({"query": "refund"})


def test_not_registered_in_the_commerce_executor():
    executor = ToolExecutor(build_commerce_tools())
    assert SEARCH_POLICY_KNOWLEDGE not in executor.names
    assert SEARCH_POLICY_KNOWLEDGE not in COMMERCE_TOOL_NAMES


@pytest.mark.parametrize(
    ("args", "query", "as_of"),
    [
        ({"query": "refund timing"}, "refund timing", None),
        ({"query": "  refund  "}, "refund", None),
        ({"query": "refund", "as_of": "2026-06-10"}, "refund", date(2026, 6, 10)),
        ({"query": "refund", "as_of": None}, "refund", None),
    ],
)
def test_valid_arguments(args, query, as_of):
    parsed = parse_policy_search_args(args)
    assert (parsed.query, parsed.as_of) == (query, as_of)


@pytest.mark.parametrize(
    ("args", "rejected"),
    [
        ({}, []),
        ({"query": ""}, []),
        ({"query": "   "}, []),
        ({"query": "x" * 501}, []),
        ({"query": 7}, []),
        ({"query": "refund", "as_of": "10/06/2026"}, []),
        ({"query": "refund", "as_of": "2026-02-30"}, []),
        ({"query": "refund", "as_of": "2026-06-10T00:00:00"}, []),
        ({"query": "refund", "as_of": 20260610}, []),
        ({"query": "refund", "tenant_id": "11111111-1111-4111-8111-111111111111"}, ["tenant_id"]),
        ({"query": "refund", "limit": 50}, ["limit"]),
        (
            {"query": "refund", "embedding_profile": "x", "runtime": {}},
            ["embedding_profile", "runtime"],
        ),
        ("refund", []),
        (None, []),
    ],
)
def test_invalid_arguments(args, rejected):
    with pytest.raises(PolicySearchArgumentError) as exc:
        parse_policy_search_args(args)
    assert exc.value.rejected == rejected
    assert "Traceback" not in str(exc.value) and len(str(exc.value)) < 200
