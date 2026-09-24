"""Deterministic final-answer grounding: citation extraction and current-run validation."""

import pytest

from app.agent.rag.grounding import check_grounding, extract_citations

V2 = "policy://delayed-shipment-compensation/v2#chunk-2"
V2_1 = "policy://delayed-shipment-compensation/v2#chunk-1"
V1 = "policy://delayed-shipment-compensation/v1#chunk-2"
CATALOG = {V2: {"citation": V2}, V2_1: {"citation": V2_1}}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No citations here.", []),
        (f"Credit is 15% [{V2}].", [V2]),
        (f"Credit ({V2}), capped [{V2_1}]; see {V2}.", [V2, V2_1]),
        (f"Ends with a period {V2}.", [V2]),
        ("Made up policy://x/v9#chunk-99, done", ["policy://x/v9#chunk-99"]),
        ("Malformed [policy://Refund Policy]", ["policy://Refund"]),
        ("policy://refund-policy/v2#chunk-1x", ["policy://refund-policy/v2#chunk-1x"]),
    ],
)
def test_extract_citations(text, expected):
    assert extract_citations(text) == expected


def check(answer, status="success", catalog=CATALOG, earlier=frozenset()):
    return check_grounding(answer, status=status, current=catalog, earlier=set(earlier))


def test_success_with_current_citation():
    outcome = check(f"15% of the order total [{V2}].")
    assert (outcome.ok, outcome.detail, outcome.citations) == (True, None, [V2])


def test_success_requires_a_citation():
    assert check("15% of the order total.").detail == "citation_required"


def test_every_citation_must_be_current():
    outcome = check(f"15% [{V2}] and 30% [{V1}].")
    assert (outcome.ok, outcome.detail) == (False, "citation_not_retrieved")


def test_previous_turn_only_citation_is_stale():
    assert check(f"30% [{V1}].", earlier={V1}).detail == "stale_citation"
    # stale even when a current citation is also present
    assert check(f"15% [{V2}], 30% [{V1}].", earlier={V1}).detail == "stale_citation"


@pytest.mark.parametrize(
    "fake",
    [
        "policy://delayed-shipment-compensation/v2#chunk-9",  # fabricated chunk
        "policy://delayed-shipment-compensation/v3#chunk-2",  # other version
        "policy://refund-policy/v2#chunk-1",  # other tenant / not retrieved
        "policy://delayed-shipment-compensation/v2",  # non-canonical
    ],
)
def test_fabricated_citations_are_not_retrieved(fake):
    assert check(f"15% [{V2}] and [{fake}].").detail == "citation_not_retrieved"


def test_no_results_answer_must_not_cite():
    assert check("No applicable policy was found.", status="no_results", catalog={}).ok
    assert check(f"See [{V2}].", status="no_results", catalog={}).detail == (
        "citation_not_retrieved"
    )


def test_invalid_retrieval_requires_a_successful_retry():
    assert check("From memory: 15%.", status="invalid", catalog={}).detail == "retrieval_required"


def test_no_retrieval_answer_without_citations_is_accepted():
    assert check("Order ORD-1001 is shipped.", status="none", catalog={}).ok


def test_no_retrieval_answer_with_citation_is_rejected():
    assert check(f"See [{V2}].", status="none", catalog={}).detail == "citation_not_retrieved"


def test_citations_used_are_deduplicated_in_order():
    outcome = check(f"a [{V2_1}] b [{V2}] c [{V2_1}]")
    assert outcome.citations == [V2_1, V2]
