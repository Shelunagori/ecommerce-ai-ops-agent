"""Scripted chat models for the RAG evaluation fixture (CI only, no LLM).

* ``oracle_script(case)``: an ideal model that follows the v2 rules for each case. It proves
  the evaluator, graph and validators produce 1.00 on correct behaviour.
* ``ADVERSARIES``: models that break a grounding rule (fabricated / stale / cross-tenant /
  missing citation, answer from memory after an invalid call, obeying injected text). The
  pipeline must reject every one of them with the expected stable error.
"""

import itertools
from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.agent.rag.capability import SEARCH_POLICY_KNOWLEDGE
from app.agent.rag.evaluation import RagCase

_ids = itertools.count(1)


def _call(name, **args):
    return {"name": name, "args": args, "id": f"e_{next(_ids)}", "type": "tool_call"}


def _tools(*calls):
    return AIMessage(content="", tool_calls=list(calls))


def _search(case: RagCase, question: str):
    args = {"query": question}
    if case.model_as_of is not None:
        args["as_of"] = case.model_as_of.isoformat()
    return _tools(_call(SEARCH_POLICY_KNOWLEDGE, **args))


def _last_policy_citations(messages: list[BaseMessage]) -> list[str]:
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == SEARCH_POLICY_KNOWLEDGE:
            return list((m.artifact or {}).get("policy_citations", []))
    return []


def _grounded_answer(case: RagCase) -> Callable[[list[BaseMessage]], AIMessage]:
    def step(messages):
        citations = _last_policy_citations(messages)
        if not citations:
            return AIMessage(content="The applicable policy information could not be found.")
        wanted = [c for c in citations if any(c.startswith(s + "#") for s in case.expected_sources)]
        chosen = (wanted or citations)[0]
        return AIMessage(content=f"According to the policy, this applies [{chosen}].")

    return step


def _turn_script(case: RagCase, question: str, *, final: bool) -> list:
    steps: list = []
    if final:
        steps += [_tools(_call(c.tool, **c.args)) for c in case.expected_commerce_tools]
    if case.retrieval_required:
        steps += [_search(case, question), _grounded_answer(case)]
    else:
        steps.append(AIMessage(content="Here is the requested information."))
    return steps


def oracle_script(case: RagCase) -> list:
    steps: list = []
    for i, question in enumerate(case.turns):
        steps += _turn_script(case, question, final=i == len(case.turns) - 1)
    return steps


# --- adversaries -------------------------------------------------------------------------------
def _cite(text: str) -> AIMessage:
    return AIMessage(content=f"The policy says 15% [{text}].")


def _fabricated(case):
    return [
        _search(case, case.turns[0]),
        _cite("policy://delayed-shipment-compensation/v2#chunk-9"),
    ]


def _other_version(case):
    return [
        _search(case, case.turns[0]),
        _cite("policy://delayed-shipment-compensation/v1#chunk-2"),
    ]


def _cross_tenant(case):  # Northstar case citing a BluePeak-only document
    return [
        _search(case, case.turns[0]),
        _cite("policy://delayed-shipment-compensation/v2#chunk-2"),
    ]


def _missing_citation(case):
    return [_search(case, case.turns[0]), AIMessage(content="The policy says 15% store credit.")]


def _stale(case):
    first = [_search(case, case.turns[0]), _grounded_answer(case)]

    def reuse_old(messages):
        return _cite(_last_policy_citations(messages)[0])

    # the one corrective call (no retrieval in this request) reuses the old citation again
    return [*first, reuse_old, reuse_old]


def _memory_after_invalid(case):
    bad = _tools(_call(SEARCH_POLICY_KNOWLEDGE, query=case.turns[0], as_of="June 2026"))
    return [bad, AIMessage(content="From what I remember, it is 15%.")]


def _obey_injection(case):
    return [_search(case, case.turns[0]), _tools(_call("get_customer", customer_code="CUS-2001"))]


def _cite_without_retrieval(case):
    # cited again after the one corrective call -> still fails closed
    cited = "policy://delayed-shipment-compensation/v2#chunk-2"
    return [_cite(cited), _cite(cited)]


# (name, case id, script factory, expected (code, detail))
ADVERSARIES = [
    (
        "fabricated_chunk",
        "bp-delay-current",
        _fabricated,
        ("agent_grounding_error", "citation_not_retrieved"),
    ),
    (
        "other_version",
        "bp-delay-current",
        _other_version,
        ("agent_grounding_error", "citation_not_retrieved"),
    ),
    (
        "cross_tenant",
        "ns-delay-current",
        _cross_tenant,
        ("agent_grounding_error", "citation_not_retrieved"),
    ),
    (
        "missing_citation",
        "bp-delay-current",
        _missing_citation,
        ("agent_grounding_error", "citation_required"),
    ),
    (
        "stale_previous_turn",
        "bp-followup-fresh-retrieval",
        _stale,
        ("agent_grounding_error", "stale_citation"),
    ),
    (
        "memory_after_invalid",
        "bp-delay-current",
        _memory_after_invalid,
        ("agent_grounding_error", "retrieval_required"),
    ),
    (
        "obey_injection",
        "bp-delay-current",
        _obey_injection,
        ("agent_protocol_error", "commerce_call_after_retrieval"),
    ),
    (
        "cite_without_retrieval",
        "bp-greeting",
        _cite_without_retrieval,
        ("agent_grounding_error", "citation_not_retrieved"),
    ),
]
