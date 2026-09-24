"""Knowledge domain on real PostgreSQL: ingestion, constraints, tenant isolation, temporal
versions, lexical retrieval and the developer CLIs. Every test runs in a rolled-back
transaction; the real synthetic corpus is ingested per test.
"""

import json
import logging
import shutil
import uuid
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import IntegrityError

from app.agent.context import AgentContext
from app.core.errors import NotFoundError
from app.knowledge.chunking import ChunkingConfig
from app.knowledge.ingest import (
    IngestionConflictError,
    chunk_id_for,
    document_id_for,
    ingest_policies,
)
from app.knowledge.limits import MAX_LIMIT
from app.knowledge.retrieval import LexicalPolicyRetriever
from app.knowledge.sources import DEFAULT_POLICY_DIR, PolicySourceError
from app.models import KnowledgeChunk, KnowledgeDocument
from app.services import KnowledgeQueries
from tests.db.conftest import FIXED_NOW

CFG = ChunkingConfig()


@pytest.fixture
def kb(db_session):
    """db_session with the real synthetic corpus ingested (rolled back after the test)."""
    report = ingest_policies(db_session, DEFAULT_POLICY_DIR, CFG)
    assert report.count("inserted") == 12
    return db_session


@pytest.fixture
def scope(kb):
    @contextmanager
    def _scope():
        yield kb

    return _scope


@pytest.fixture
def retriever(scope):
    return LexicalPolicyRetriever(scope, clock=lambda: FIXED_NOW)


@pytest.fixture
def ctx_a(tenant_a):
    return AgentContext(tenant_a.tenant_id, "req-a")


@pytest.fixture
def ctx_b(tenant_b):
    return AgentContext(tenant_b.tenant_id, "req-b")


def copy_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "policies"
    shutil.copytree(DEFAULT_POLICY_DIR, root)
    return root


def counts(session) -> tuple[int, int]:
    return (
        session.scalar(select(func.count()).select_from(KnowledgeDocument)),
        session.scalar(select(func.count()).select_from(KnowledgeChunk)),
    )


def snapshot(session) -> list[tuple]:
    rows = session.execute(
        select(
            KnowledgeDocument.id,
            KnowledgeDocument.immutable_content_hash,
            KnowledgeDocument.effective_to,
            KnowledgeDocument.updated_at,
        ).order_by(KnowledgeDocument.id)
    ).all()
    chunks = session.execute(
        select(KnowledgeChunk.id, KnowledgeChunk.content).order_by(KnowledgeChunk.id)
    ).all()
    return [tuple(r) for r in rows] + [tuple(c) for c in chunks]


# --- ingestion -------------------------------------------------------------------------------
def test_ingestion_is_idempotent(kb):
    assert counts(kb) == (12, 51)
    before = snapshot(kb)
    report = ingest_policies(kb, DEFAULT_POLICY_DIR, CFG)
    assert report.count("unchanged") == 12 and report.conflicts == []
    assert counts(kb) == (12, 51) and snapshot(kb) == before


def test_ids_are_deterministic_and_tenant_scoped(kb, tenant_a, tenant_b):
    for tenant in (tenant_a, tenant_b):
        doc_id = document_id_for(tenant.tenant_id, "refund-policy", 1)
        doc = kb.get(KnowledgeDocument, doc_id)
        assert doc is not None and doc.tenant_id == tenant.tenant_id
        for chunk in kb.scalars(select(KnowledgeChunk).where(KnowledgeChunk.document_id == doc_id)):
            assert chunk.id == chunk_id_for(doc_id, chunk.chunk_index, chunk.content_hash)
    assert document_id_for(tenant_a.tenant_id, "refund-policy", 1) != document_id_for(
        tenant_b.tenant_id, "refund-policy", 1
    )


def test_stored_provenance(kb, tenant_a):
    doc = kb.get(KnowledgeDocument, document_id_for(tenant_a.tenant_id, "refund-policy", 2))
    assert (doc.chunker, doc.chunking_hash, doc.chunk_count) == (
        "policy-section-v1",
        CFG.chunking_hash,
        4,
    )
    assert doc.source_name == "northstar-commerce/refund-policy.v2.md"
    assert (doc.effective_from, doc.effective_to) == (date(2026, 7, 1), None)


def test_changed_source_under_an_existing_version_is_refused_atomically(kb, tmp_path):
    root = copy_corpus(tmp_path)
    changed = root / "northstar-commerce" / "refund-policy.v1.md"
    changed.write_text(changed.read_text().replace("10 business days", "8 business days"))
    # A legitimately new document in the same run must NOT be written either.
    (root / "bluepeak-retail" / "warranty-policy.v1.md").write_text(
        "---\ntenant: bluepeak-retail\ndocument_key: warranty-policy\ntitle: Warranty\n"
        "document_type: policy\nversion: 1\neffective_from: 2026-02-01\neffective_to:\n---\n"
        "# Warranty\n\n## Scope\n\nFictional two-year warranty.\n"
    )
    before = snapshot(kb)
    with pytest.raises(IngestionConflictError) as exc:
        ingest_policies(kb, root, CFG)
    outcomes = {
        (d.tenant, d.document_key, d.version): d.outcome for d in exc.value.report.documents
    }
    assert outcomes[("northstar-commerce", "refund-policy", 1)] == "source_conflict"
    assert outcomes[("bluepeak-retail", "warranty-policy", 1)] == "inserted"
    assert snapshot(kb) == before and counts(kb) == (12, 51)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("10 business days", "8 business days"),  # body
        ("title: Refund Policy", "title: Refund Rules"),  # immutable metadata
        ("effective_from: 2026-01-01", "effective_from: 2026-01-02"),  # effective_from
    ],
    ids=["body", "title", "effective_from"],
)
def test_immutable_payload_changes_are_source_conflicts(kb, tmp_path, old, new):
    root = copy_corpus(tmp_path)
    path = root / "northstar-commerce" / "refund-policy.v1.md"
    assert old in path.read_text()
    path.write_text(path.read_text().replace(old, new, 1))
    before = snapshot(kb)
    with pytest.raises(IngestionConflictError) as exc:
        ingest_policies(kb, root, CFG)
    [conflict] = exc.value.report.conflicts
    assert (conflict.document_key, conflict.version, conflict.outcome) == (
        "refund-policy",
        1,
        "source_conflict",
    )
    assert snapshot(kb) == before


def test_a_set_end_date_cannot_be_removed(kb, tmp_path):
    root = copy_corpus(tmp_path)
    extra = root / "bluepeak-retail" / "withdrawn-policy.v1.md"
    extra.write_text(
        "---\ntenant: bluepeak-retail\ndocument_key: withdrawn-policy\ntitle: Withdrawn\n"
        "document_type: policy\nversion: 1\neffective_from: 2026-02-01\n"
        "effective_to: 2026-05-01\n---\n# Withdrawn\n\n## Scope\n\nFictional.\n"
    )
    assert ingest_policies(kb, root, CFG).count("inserted") == 1
    extra.write_text(extra.read_text().replace("effective_to: 2026-05-01", "effective_to:"))
    with pytest.raises(IngestionConflictError) as exc:
        ingest_policies(kb, root, CFG)
    assert [(d.document_key, d.outcome) for d in exc.value.report.conflicts] == [
        ("withdrawn-policy", "source_conflict")
    ]


def test_changed_chunking_configuration_is_a_mismatch_not_a_source_change(kb):
    before = snapshot(kb)
    with pytest.raises(IngestionConflictError) as exc:
        ingest_policies(kb, DEFAULT_POLICY_DIR, ChunkingConfig(max_chars=800))
    assert {d.outcome for d in exc.value.report.documents} == {"chunking_mismatch"}
    assert snapshot(kb) == before


def test_new_version_is_inserted_and_its_predecessor_retired_once(kb, tmp_path, tenant_a):
    root = copy_corpus(tmp_path)
    v2 = root / "northstar-commerce" / "refund-policy.v2.md"
    v2.write_text(v2.read_text().replace("effective_to:\n", "effective_to: 2026-10-01\n"))
    v3_text = (
        v2.read_text()
        .replace("version: 2", "version: 3")
        .replace("effective_from: 2026-07-01", "effective_from: 2026-10-01")
        .replace("effective_to: 2026-10-01\n", "effective_to:\n")
        .replace("5 business days", "3 business days")
    )
    (root / "northstar-commerce" / "refund-policy.v3.md").write_text(v3_text)

    report = ingest_policies(kb, root, CFG)
    outcomes = {
        (d.document_key, d.version): d.outcome
        for d in report.documents
        if d.tenant == "northstar-commerce"
    }
    assert outcomes[("refund-policy", 2)] == "retired"
    assert outcomes[("refund-policy", 3)] == "inserted"
    assert report.count("unchanged") == 11 and counts(kb) == (13, 55)
    q = KnowledgeQueries(kb, tenant_a)
    assert q.get_effective_document("refund-policy", date(2026, 9, 30)).version == 2
    assert q.get_effective_document("refund-policy", date(2026, 10, 1)).version == 3
    assert "5 business days" in q.list_chunks("refund-policy", 2)[1].content  # v2 unchanged

    # The end date may be set once; changing it later (here: earlier, leaving a gap, so the
    # ranges stay valid) is a conflict with the stored, immutable version.
    v2.write_text(v2.read_text().replace("effective_to: 2026-10-01", "effective_to: 2026-09-01"))
    with pytest.raises(IngestionConflictError) as exc:
        ingest_policies(kb, root, CFG)
    assert [d.outcome for d in exc.value.report.conflicts] == ["source_conflict"]


def test_overlap_with_a_stored_version_is_rejected(kb, tmp_path):
    root = copy_corpus(tmp_path)
    (root / "northstar-commerce" / "refund-policy.v2.md").unlink()  # stored v2 stays open
    (root / "northstar-commerce" / "refund-policy.v3.md").write_text(
        "---\ntenant: northstar-commerce\ndocument_key: refund-policy\ntitle: Refund Policy\n"
        "document_type: policy\nversion: 3\neffective_from: 2026-10-01\neffective_to:\n---\n"
        "# Refund Policy\n\n## Timing\n\nFictional.\n"
    )
    before = snapshot(kb)
    with pytest.raises(PolicySourceError, match="overlaps"):
        ingest_policies(kb, root, CFG)
    assert snapshot(kb) == before


def test_ingestion_never_deletes_rows_whose_files_are_gone(kb, tmp_path):
    root = copy_corpus(tmp_path)
    (root / "bluepeak-retail" / "returns-policy.v1.md").unlink()
    report = ingest_policies(kb, root, CFG)
    assert report.count("unchanged") == 11 and counts(kb) == (12, 51)


def test_unknown_tenant_directory_fails_before_writing(kb, tmp_path):
    root = copy_corpus(tmp_path)
    (root / "acme-shop").mkdir()
    (root / "acme-shop" / "refund-policy.v1.md").write_text(
        (root / "northstar-commerce" / "refund-policy.v1.md")
        .read_text()
        .replace("northstar-commerce", "acme-shop")
    )
    with pytest.raises(PolicySourceError, match="unknown tenant"):
        ingest_policies(kb, root, CFG)


# --- database constraints ----------------------------------------------------------------------
def _doc(tenant_id, key="refund-policy", version=1, start=date(2026, 1, 1), end=None, **kw):
    return KnowledgeDocument(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        document_key=key,
        title="T",
        document_type="policy",
        version=version,
        effective_from=start,
        effective_to=end,
        source_name="x.md",
        immutable_content_hash="a" * 64,
        chunker="policy-section-v1",
        chunking_hash="b" * 64,
        chunk_count=1,
        **kw,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"version": 0},
        {"version": -2},
        {"end": date(2026, 1, 1)},
        {"end": date(2025, 12, 31)},
        {"key": "Bad Key"},
        {"document_type": "memo"},
    ],
)
def test_document_check_constraints(db_session, tenant_a, kwargs):
    extra = {"document_type": kwargs.pop("document_type")} if "document_type" in kwargs else {}
    doc = _doc(tenant_a.tenant_id, **kwargs)
    for k, v in extra.items():
        setattr(doc, k, v)
    db_session.add(doc)
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_version_uniqueness_is_scoped_to_the_tenant(db_session, tenant_a, tenant_b):
    db_session.add_all(
        [
            _doc(tenant_a.tenant_id, end=date(2026, 3, 1)),
            _doc(tenant_b.tenant_id, end=date(2026, 3, 1)),
        ]
    )
    db_session.flush()  # same key+version in two tenants is fine
    db_session.add(_doc(tenant_a.tenant_id, start=date(2026, 5, 1)))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_only_one_open_ended_version_per_document(db_session, tenant_a):
    db_session.add(_doc(tenant_a.tenant_id, key="x-policy", version=1))
    db_session.flush()
    db_session.add(_doc(tenant_a.tenant_id, key="x-policy", version=2, start=date(2026, 6, 1)))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_database_alone_does_not_prevent_every_closed_range_overlap(db_session, tenant_a):
    """Documented limit: two CLOSED overlapping ranges pass the database; the validated
    ingestion path is what keeps ranges disjoint (see test_version_ranges_must_not_overlap)."""
    db_session.add_all(
        [
            _doc(tenant_a.tenant_id, key="y-policy", version=1, end=date(2026, 6, 1)),
            _doc(
                tenant_a.tenant_id,
                key="y-policy",
                version=2,
                start=date(2026, 3, 1),
                end=date(2026, 9, 1),
            ),
        ]
    )
    db_session.flush()


def test_chunk_cannot_reference_another_tenants_document(kb, tenant_a, tenant_b):
    b_doc = document_id_for(tenant_b.tenant_id, "refund-policy", 1)
    kb.add(
        KnowledgeChunk(
            id=uuid.uuid4(),
            tenant_id=tenant_a.tenant_id,
            document_id=b_doc,
            chunk_index=99,
            section="S",
            content="x",
            char_count=1,
            content_hash="c" * 64,
        )
    )
    with pytest.raises(IntegrityError):
        kb.flush()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_index": -1},
        {"char_count": 5},
        {"content": "", "char_count": 0},
        {"content_hash": "nothex"},
    ],
)
def test_chunk_check_constraints(kb, tenant_a, kwargs):
    doc = document_id_for(tenant_a.tenant_id, "refund-policy", 1)
    values = {
        "chunk_index": 50,
        "section": "S",
        "content": "abc",
        "char_count": 3,
        "content_hash": "d" * 64,
        **kwargs,
    }
    kb.add(KnowledgeChunk(id=uuid.uuid4(), tenant_id=tenant_a.tenant_id, document_id=doc, **values))
    with pytest.raises(IntegrityError):
        kb.flush()


def test_duplicate_chunk_index_is_rejected(kb, tenant_a):
    doc = document_id_for(tenant_a.tenant_id, "refund-policy", 1)
    kb.add(
        KnowledgeChunk(
            id=uuid.uuid4(),
            tenant_id=tenant_a.tenant_id,
            document_id=doc,
            chunk_index=0,
            section="S",
            content="x",
            char_count=1,
            content_hash="e" * 64,
        )
    )
    with pytest.raises(IntegrityError):
        kb.flush()


# --- tenant-scoped queries and temporal versions -------------------------------------------------
def test_list_documents_is_tenant_scoped(kb, tenant_a, tenant_b):
    a = KnowledgeQueries(kb, tenant_a).list_documents()
    b = KnowledgeQueries(kb, tenant_b).list_documents()
    assert len(a) == len(b) == 6
    assert ("refund-policy", 2) in [(d.document_key, d.version) for d in a]
    assert ("refund-policy", 2) not in [(d.document_key, d.version) for d in b]


@pytest.mark.parametrize(
    ("as_of", "version"),
    [
        (date(2025, 12, 31), None),  # before the first version
        (date(2026, 1, 1), 1),  # effective_from is inclusive
        (date(2026, 6, 10), 1),
        (date(2026, 6, 30), 1),  # last day of v1
        (date(2026, 7, 1), 2),  # v1.effective_to is exclusive -> v2
        (date(2026, 9, 23), 2),
        (datetime(2026, 7, 1, 8, 0, tzinfo=timezone(timedelta(hours=10))), 1),  # 06-30 UTC
        (datetime(2026, 6, 30, 22, 0, tzinfo=timezone(timedelta(hours=-5))), 2),  # 07-01 UTC
        (datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC), 1),
    ],
)
def test_northstar_refund_policy_versions(kb, tenant_a, as_of, version):
    doc = KnowledgeQueries(kb, tenant_a).get_effective_document("refund-policy", as_of)
    assert (doc.version if doc else None) == version


@pytest.mark.parametrize(
    ("as_of", "version"),
    [
        (date(2026, 2, 28), None),
        (date(2026, 3, 1), 1),
        (date(2026, 8, 14), 1),
        (date(2026, 8, 15), 2),
        (date(2027, 1, 1), 2),
    ],
)
def test_bluepeak_delay_compensation_versions(kb, tenant_b, as_of, version):
    doc = KnowledgeQueries(kb, tenant_b).get_effective_document(
        "delayed-shipment-compensation", as_of
    )
    assert (doc.version if doc else None) == version


def test_effective_versions_never_overlap_in_the_corpus(kb, tenant_a, tenant_b):
    for tenant in (tenant_a, tenant_b):
        q = KnowledgeQueries(kb, tenant)
        keys = {d.document_key for d in q.list_documents()}
        day = date(2025, 12, 1)
        while day <= date(2027, 1, 31):
            for key in keys:
                matches = [
                    d
                    for d in q.list_documents()
                    if d.document_key == key
                    and d.effective_from <= day
                    and (d.effective_to is None or day < d.effective_to)
                ]
                assert len(matches) <= 1
            day += timedelta(days=15)


def test_naive_as_of_is_rejected(kb, tenant_a):
    with pytest.raises(ValueError):
        KnowledgeQueries(kb, tenant_a).get_effective_document("refund-policy", datetime(2026, 7, 1))


def test_other_tenant_has_no_northstar_only_version(kb, tenant_b):
    with pytest.raises(NotFoundError):
        KnowledgeQueries(kb, tenant_b).get_document("refund-policy", 2)


def test_chunk_lookup_by_id_and_citation_is_tenant_bound(kb, tenant_a, tenant_b):
    qa, qb = KnowledgeQueries(kb, tenant_a), KnowledgeQueries(kb, tenant_b)
    a_chunk = qa.list_chunks("returns-policy", 1)[1]
    assert qa.get_chunk(a_chunk.chunk_id) == a_chunk
    with pytest.raises(NotFoundError):
        qb.get_chunk(a_chunk.chunk_id)  # another tenant's chunk id is simply not found
    # The same tenant-relative citation resolves to each tenant's own text.
    cite = "policy://returns-policy/v1#chunk-1"
    assert (
        qa.get_chunk_by_citation(cite).content
        == "Items can be returned within 30 days of delivery."
    )
    assert (
        qb.get_chunk_by_citation(cite).content
        == "Items can be returned within 14 days of delivery."
    )
    with pytest.raises(NotFoundError):
        qb.get_chunk_by_citation("policy://refund-policy/v2#chunk-1")
    for bad in ("policy://x/v1#chunk-1; DROP TABLE t", "/data/policies/x.md"):
        with pytest.raises(NotFoundError):
            qa.get_chunk_by_citation(bad)


def test_chunks_carry_section_and_no_tenant_or_path(kb, tenant_a):
    chunks = KnowledgeQueries(kb, tenant_a).list_chunks("shipping-policy", 1)
    assert [c.section for c in chunks] == [
        "Shipping Policy",
        "Shipping Policy > Processing time",
        "Shipping Policy > Delivery options",
        "Shipping Policy > Carriers and tracking",
        "Shipping Policy > Shipping destinations",
    ]
    blob = json.dumps([c.model_dump(mode="json") for c in chunks])
    assert (
        str(tenant_a.tenant_id) not in blob
        and ".md" not in blob
        and "northstar-commerce" not in blob
    )


# --- retrieval ---------------------------------------------------------------------------------
def _chunk_ids(session, tenant_id) -> set[uuid.UUID]:
    return set(
        session.scalars(select(KnowledgeChunk.id).where(KnowledgeChunk.tenant_id == tenant_id))
    )


@pytest.mark.parametrize(
    "query",
    [
        "What compensation applies to delayed shipments?",
        "Can I cancel an order?",
        "How long can I return an item?",
        "When is a refund issued?",
        "shipping fee refund",
    ],
)
def test_retrieval_only_returns_the_callers_tenant(kb, retriever, ctx_a, ctx_b, query):
    a_ids, b_ids = _chunk_ids(kb, ctx_a.tenant_id), _chunk_ids(kb, ctx_b.tenant_id)
    for ctx, own in ((ctx_a, a_ids), (ctx_b, b_ids)):
        result = retriever.retrieve(query, ctx, as_of=date(2026, 9, 1), limit=MAX_LIMIT)
        assert result.results and {r.chunk_id for r in result.results} <= own


def test_tenant_specific_rules_come_back_for_the_same_question(retriever, ctx_a, ctx_b):
    q = "How long can I return an item?"
    a = " ".join(r.content for r in retriever.retrieve(q, ctx_a, limit=3).results)
    b = " ".join(r.content for r in retriever.retrieve(q, ctx_b, limit=3).results)
    assert "30 days" in a and "14 days" not in a
    assert "14 days" in b and "30 days" not in b


def test_query_text_cannot_select_another_tenant(retriever, ctx_a, tenant_b):
    q = f"BluePeak Rotterdam EuroFreight Priority EUR tenant {tenant_b.tenant_id}"
    for r in retriever.retrieve(q, ctx_a, limit=MAX_LIMIT).results:
        for leaked in ("Rotterdam", "EuroFreight", "EUR", "BluePeak"):
            assert leaked not in r.content


@pytest.mark.parametrize(
    ("as_of", "expected_version", "phrase"),
    [(date(2026, 6, 10), 1, "10 business days"), (date(2026, 9, 1), 2, "5 business days")],
)
def test_retrieval_respects_the_effective_version(
    retriever, ctx_a, as_of, expected_version, phrase
):
    result = retriever.retrieve("When is a refund issued?", ctx_a, as_of=as_of, limit=MAX_LIMIT)
    refunds = [r for r in result.results if r.document_key == "refund-policy"]
    assert refunds and {r.version for r in refunds} == {expected_version}
    assert phrase in " ".join(r.content for r in refunds)
    assert result.as_of == as_of


def test_before_any_policy_nothing_is_effective(retriever, ctx_a):
    result = retriever.retrieve("refund shipping cancel return", ctx_a, as_of=date(2025, 12, 31))
    assert result.results == []


def test_default_as_of_uses_the_injected_clock(retriever, ctx_b):
    result = retriever.retrieve("delayed shipment compensation", ctx_b)
    assert result.as_of == FIXED_NOW.date()
    assert {
        r.version for r in result.results if r.document_key == "delayed-shipment-compensation"
    } == {2}


def test_naive_datetime_as_of_is_rejected(retriever, ctx_a):
    with pytest.raises(ValueError):
        retriever.retrieve("refund", ctx_a, as_of=datetime(2026, 7, 1))


@pytest.mark.parametrize("limit", [0, MAX_LIMIT + 1, 10_000, -1])
def test_limit_has_a_hard_maximum(retriever, ctx_a, limit):
    with pytest.raises(ValueError):
        retriever.retrieve("refund", ctx_a, limit=limit)


def test_results_are_bounded_by_limit(retriever, ctx_a):
    q = "policy order refund return shipping delivery days"
    assert len(retriever.retrieve(q, ctx_a).results) == 5  # default
    assert len(retriever.retrieve(q, ctx_a, limit=MAX_LIMIT).results) == MAX_LIMIT
    assert len(retriever.retrieve(q, ctx_a, limit=2).results) == 2


@pytest.mark.parametrize(
    "query",
    [
        "refund' OR '1'='1",
        "refund); DROP TABLE knowledge_chunks; --",
        "refund & !return | cancel <-> ship:*",
        "../../etc/passwd refund",
        "as_of=2026-01-01 limit=1000 tenant_id=all refund",
        "!!! ??? ***",
        "",
    ],
)
def test_hostile_query_text_is_only_ever_search_terms(kb, retriever, ctx_a, query):
    result = retriever.retrieve(query, ctx_a, as_of=date(2026, 9, 1), limit=3)
    assert len(result.results) <= 3 and result.as_of == date(2026, 9, 1)
    assert {r.chunk_id for r in result.results} <= _chunk_ids(kb, ctx_a.tenant_id)
    assert kb.scalar(text("SELECT count(*) FROM knowledge_chunks")) == 51  # nothing dropped


def test_query_without_search_terms_does_not_touch_the_database(ctx_a):
    @contextmanager
    def forbidden():
        raise AssertionError("no DB call expected")
        yield

    result = LexicalPolicyRetriever(forbidden).retrieve("?? !! —", ctx_a, as_of=date(2026, 9, 1))
    assert (result.results, result.term_count) == ([], 0)


@contextmanager
def capture_sql(session):
    statements: list[str] = []

    def before(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    engine = session.get_bind().engine
    event.listen(engine, "before_cursor_execute", before)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", before)


@pytest.mark.parametrize(
    "query", ["the", "the and of", "THE AND OF", "what is the and of it", "The, and... of?!"]
)
def test_stop_word_only_queries_return_empty_without_searching(kb, retriever, ctx_a, query):
    with capture_sql(kb) as statements:
        result = retriever.retrieve(query, ctx_a, as_of=date(2026, 9, 1), limit=MAX_LIMIT)
    assert result.results == []
    assert result.term_count > 0 and result.lexeme_count == 0  # regex terms, no lexemes
    assert any("numnode" in s for s in statements)  # the pre-check ran...
    assert not any("ts_rank_cd" in s for s in statements)  # ...and the search never did
    assert result == retriever.retrieve(query, ctx_a, as_of=date(2026, 9, 1), limit=MAX_LIMIT)


def test_stop_words_do_not_hide_a_meaningful_term(retriever, ctx_a):
    result = retriever.retrieve("the and of refund", ctx_a, as_of=date(2026, 9, 1))
    assert result.lexeme_count == 1 and result.results
    assert all(r.score > 0 for r in result.results)


def test_ranking_is_deterministic_and_totally_ordered(retriever, ctx_a):
    q = "refund shipping days"
    first = retriever.retrieve(q, ctx_a, limit=MAX_LIMIT)
    assert first == retriever.retrieve(q, ctx_a, limit=MAX_LIMIT)
    keys = [(-r.score, r.document_key, -r.version, r.chunk_index) for r in first.results]
    assert keys == sorted(keys)
    assert [r.rank for r in first.results] == list(range(1, len(first.results) + 1))


def test_title_is_part_of_the_ranking_vector(kb, retriever, ctx_a):
    """A word that appears ONLY in the document title (not in section or body) matches."""
    doc = _doc(ctx_a.tenant_id, key="giftcard-policy", end=None)
    doc.title = "Giftcard Policy"
    kb.add(doc)
    kb.flush()
    kb.add(
        KnowledgeChunk(
            id=uuid.uuid4(),
            tenant_id=ctx_a.tenant_id,
            document_id=doc.id,
            chunk_index=0,
            section="Balance",
            content="Balances never expire.",
            char_count=22,
            content_hash="f" * 64,
        )
    )
    kb.flush()
    [hit] = retriever.retrieve("giftcard", ctx_a, as_of=date(2026, 9, 1)).results
    assert (hit.document_key, hit.section) == ("giftcard-policy", "Balance")


def test_title_and_section_weigh_more_than_body(retriever, ctx_a):
    # "cancellation" is in the Order Cancellation title/sections and in one refund-policy line.
    top = retriever.retrieve("cancellation", ctx_a, limit=3).results[0]
    assert top.document_key == "order-cancellation"


def test_result_contract_and_citations(retriever, ctx_a):
    [r] = retriever.retrieve("PO boxes", ctx_a, as_of=date(2026, 9, 1), limit=1).results
    assert r.citation == "policy://shipping-policy/v1#chunk-4"
    assert (r.document_key, r.title, r.version, r.section, r.chunk_index) == (
        "shipping-policy",
        "Shipping Policy",
        1,
        "Shipping Policy > Shipping destinations",
        4,
    )
    assert (r.effective_from, r.effective_to) == (date(2026, 1, 1), None)
    assert "PO boxes" in r.content and r.score > 0 and r.rank == 1
    dumped = json.dumps(r.model_dump(mode="json"))
    for leak in (str(ctx_a.tenant_id), "northstar-commerce", ".md", "data/policies", "file:"):
        assert leak not in dumped


def test_retrieval_log_is_safe(retriever, ctx_a, caplog):
    with caplog.at_level(logging.INFO, logger="app.knowledge.retrieval"):
        retriever.retrieve("PRIVATE-QUERY-TEXT refund timing", ctx_a, as_of=date(2026, 9, 1))
    [rec] = [r for r in caplog.records if r.name == "app.knowledge.retrieval"]
    assert (rec.retriever, rec.result_count, rec.request_id) == ("lexical-pg-fts-v1", 5, "req-a")
    assert rec.citations and rec.query_chars == len("PRIVATE-QUERY-TEXT refund timing")
    assert "PRIVATE-QUERY-TEXT" not in caplog.text and "business days" not in caplog.text


def test_ingestion_log_is_safe(db_session, caplog):
    with caplog.at_level(logging.INFO, logger="app.knowledge.ingest"):
        ingest_policies(db_session, DEFAULT_POLICY_DIR, CFG)
    recs = [r for r in caplog.records if r.name == "app.knowledge.ingest"]
    assert len(recs) == 12
    assert {(r.outcome, r.chunk_count > 0) for r in recs} == {("inserted", True)}
    assert all(r.tenant_id and r.document_key and r.version for r in recs)
    assert "business days" not in caplog.text and "Fictional" not in caplog.text


def test_context_must_be_agent_context(retriever, tenant_a):
    with pytest.raises(TypeError):
        retriever.retrieve("refund", {"tenant_id": str(tenant_a.tenant_id)})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("ctx_name", "key", "boundary", "after"),
    [
        ("ctx_a", "refund-policy", date(2026, 7, 1), 2),
        ("ctx_b", "delayed-shipment-compensation", date(2026, 8, 15), 2),
    ],
)
def test_retrieval_on_the_exact_boundary_day_sees_only_the_new_version(
    request, retriever, ctx_name, key, boundary, after
):
    ctx = request.getfixturevalue(ctx_name)
    words = key.replace("-", " ")
    for day, version in ((boundary - timedelta(days=1), after - 1), (boundary, after)):
        results = retriever.retrieve(words, ctx, as_of=day, limit=MAX_LIMIT).results
        assert {r.version for r in results if r.document_key == key} == {version}


def test_effective_to_is_exclusive_even_without_a_successor(db_session, tenant_a):
    db_session.add(_doc(tenant_a.tenant_id, key="withdrawn-policy", end=date(2026, 3, 1)))
    db_session.flush()
    q = KnowledgeQueries(db_session, tenant_a)
    assert q.get_effective_document("withdrawn-policy", date(2026, 2, 28)).version == 1
    assert q.get_effective_document("withdrawn-policy", date(2026, 3, 1)) is None
