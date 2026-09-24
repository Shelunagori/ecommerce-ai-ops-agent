"""Temporal semantics, citations, query-term extraction and limits. No DB."""

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from app.knowledge.citations import citation_for, parse_citation
from app.knowledge.limits import MAX_LIMIT, validate_limit
from app.knowledge.retrieval import MAX_QUERY_TERMS, query_terms
from app.knowledge.temporal import effective_date


# --- effective_date ---------------------------------------------------------------------------
def test_dates_are_used_directly():
    assert effective_date(date(2026, 6, 30)) == date(2026, 6, 30)


def test_aware_datetimes_use_their_utc_calendar_date():
    plus10 = timezone(timedelta(hours=10))
    minus5 = timezone(timedelta(hours=-5))
    # 2026-07-01 08:00 +10:00 is still 2026-06-30 in UTC.
    assert effective_date(datetime(2026, 7, 1, 8, 0, tzinfo=plus10)) == date(2026, 6, 30)
    # 2026-06-30 21:00 -05:00 is already 2026-07-01 in UTC.
    assert effective_date(datetime(2026, 6, 30, 21, 0, tzinfo=minus5)) == date(2026, 7, 1)
    assert effective_date(datetime(2026, 7, 1, 0, 0, tzinfo=UTC)) == date(2026, 7, 1)


def test_naive_datetimes_are_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        effective_date(datetime(2026, 7, 1, 12, 0))


@pytest.mark.parametrize("bad", ["2026-07-01", 20260701, None])
def test_other_types_are_rejected(bad):
    with pytest.raises(TypeError):
        effective_date(bad)  # type: ignore[arg-type]


# --- citations --------------------------------------------------------------------------------
def test_citation_round_trip_is_tenant_free():
    c = citation_for("delayed-shipment-compensation", 2, 3)
    assert c == "policy://delayed-shipment-compensation/v2#chunk-3"
    ref = parse_citation(c)
    assert (ref.document_key, ref.version, ref.chunk_index) == (
        "delayed-shipment-compensation",
        2,
        3,
    )


@pytest.mark.parametrize(
    "bad",
    [
        "policy://refund-policy/v0#chunk-1",
        "policy://refund-policy/v1#chunk--1",
        "policy://../etc/passwd/v1#chunk-1",
        "policy://northstar-commerce/refund-policy/v1#chunk-1",  # no tenant segment
        "file:///data/policies/northstar-commerce/refund-policy.v1.md",
        "policy://refund-policy/v1#chunk-1 OR 1=1",
        "",
    ],
)
def test_invalid_citations_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_citation(bad)


# --- untrusted query text ----------------------------------------------------------------------
def test_query_terms_drop_operators_punctuation_and_sql():
    terms = query_terms('Refund\' OR 1=1; DROP TABLE knowledge_chunks; -- & | ! <-> :* "x"')
    assert terms == ["refund", "1", "drop", "table", "knowledge", "chunks", "x"]
    assert all(t.isalnum() and t == t.lower() for t in terms)


def test_query_terms_are_bounded_and_deduplicated():
    assert query_terms("Ship ship SHIP shipping") == ["ship", "shipping"]
    many = " ".join(f"w{i}" for i in range(100))
    assert len(query_terms(many)) == MAX_QUERY_TERMS
    long_tail = "a " * 250 + "zzztail"
    assert "zzztail" not in query_terms(long_tail)  # beyond the 500-char cap


@pytest.mark.parametrize("text", ["", "   ", "?!", "— … ¿", "tenant_id="])
def test_queries_without_letters_or_digits(text):
    assert query_terms(text) in ([], ["tenant", "id"])


def test_query_must_be_text():
    with pytest.raises(TypeError):
        query_terms(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [0, -1, MAX_LIMIT + 1, 1000, True, 2.0, "5"])
def test_limit_bounds(limit):
    with pytest.raises(ValueError):
        validate_limit(limit)  # type: ignore[arg-type]


def test_limit_defaults_and_maximum():
    assert validate_limit(1) == 1 and validate_limit(MAX_LIMIT) == MAX_LIMIT == 10


def test_one_authoritative_limit_definition():
    """lexical, semantic, the vector query boundary and the CLI all use app.knowledge.limits."""
    import inspect

    import app.knowledge.retrieval as lexical
    import app.knowledge.semantic as semantic
    import app.services.knowledge as service
    import scripts.search_policies as cli
    from app.knowledge import limits

    assert (limits.DEFAULT_LIMIT, limits.MAX_LIMIT) == (5, 10)
    assert lexical.validate_limit is semantic.validate_limit is service.validate_limit
    assert lexical.validate_limit is limits.validate_limit and cli.MAX_LIMIT == limits.MAX_LIMIT
    for module in (lexical, semantic, service):
        source = inspect.getsource(module)
        for literal in ("<= 10", "MAX_LIMIT = ", "DEFAULT_LIMIT = "):
            assert literal not in source, (module.__name__, literal)
