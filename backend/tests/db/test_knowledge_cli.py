"""Developer CLIs: scripts.ingest_policies and scripts.search_policies (real PostgreSQL).

Both run inside the rolled-back test transaction (their session factories are patched)."""

import json
import uuid
from contextlib import contextmanager

import pytest

import app.db.session as db_session_module
from app.core.config import get_settings
from scripts import ingest_policies as ingest_cli
from scripts import search_policies as search_cli


@pytest.fixture
def same_session(db_session, monkeypatch):
    @contextmanager
    def scope():
        yield db_session

    monkeypatch.setattr(db_session_module, "unit_of_work", scope)
    monkeypatch.setattr(search_cli, "read_only_session", scope)
    return db_session


def test_ingest_cli_runs_twice_idempotently(same_session, capsys):
    assert ingest_cli.main([]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "ok" and first["counts"]["inserted"] == 12
    assert first["chunker"] == "policy-section-v1"
    assert ingest_cli.main([]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["counts"] == {
        "inserted": 0,
        "unchanged": 12,
        "retired": 0,
        "source_conflict": 0,
        "chunking_mismatch": 0,
    }
    assert all("source" not in d for d in second["documents"])  # no file paths


def test_ingest_cli_reports_conflicts_without_writing(same_session, capsys, monkeypatch):
    assert ingest_cli.main([]) == 0
    capsys.readouterr()
    monkeypatch.setenv("KNOWLEDGE_CHUNK_MAX_CHARS", "800")
    get_settings.cache_clear()
    try:
        assert ingest_cli.main([]) == 1
    finally:
        get_settings.cache_clear()
    err = capsys.readouterr().err
    summary = json.loads(err[err.index("{\n") :])
    assert summary["status"] == "refused" and summary["counts"]["chunking_mismatch"] == 12


def test_ingest_cli_refuses_production(monkeypatch, capsys):
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    try:
        assert ingest_cli.main([]) == 2
    finally:
        get_settings.cache_clear()
    assert "Refusing" in capsys.readouterr().err


def test_search_cli_outputs_tenant_safe_json(same_session, tenant_a, tenant_b, capsys):
    assert ingest_cli.main([]) == 0
    capsys.readouterr()
    outputs = {}
    for name, tenant in (("ns", tenant_a), ("bp", tenant_b)):
        code = search_cli.main(
            [
                "--tenant",
                str(tenant.tenant_id),
                "--query",
                "compensation for delayed shipment",
                "--as-of",
                "2026-09-01",
                "--limit",
                "3",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        outputs[name] = json.loads(out)
        assert str(tenant.tenant_id) not in out and ".md" not in out
    ns, bp = outputs["ns"], outputs["bp"]
    assert ns["query"] == "compensation for delayed shipment" and ns["as_of"] == "2026-09-01"
    assert ns["retriever"] == "lexical-pg-fts-v1" and len(ns["results"]) == 3
    top_ns, top_bp = ns["results"][0], bp["results"][0]
    assert top_ns["citation"].startswith("policy://delayed-shipment-compensation/v1#")
    assert top_bp["citation"].startswith("policy://delayed-shipment-compensation/v2#")
    assert {"score", "version", "effective_from", "effective_to", "section"} <= set(top_ns)


def test_search_cli_historical_lookup(same_session, tenant_b, capsys):
    assert ingest_cli.main([]) == 0
    capsys.readouterr()
    assert (
        search_cli.main(
            [
                "--tenant",
                str(tenant_b.tenant_id),
                "--query",
                "delayed shipment compensation",
                "--as-of",
                "2026-06-10",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    versions = {
        r["version"] for r in out["results"] if r["document_key"] == "delayed-shipment-compensation"
    }
    assert versions == {1}


@pytest.mark.parametrize(
    "args",
    [
        ["--as-of", "2026-06-10T12:00:00"],  # naive datetime
        ["--limit", "11"],
        ["--limit", "0"],
        ["--as-of", "yesterday"],
    ],
)
def test_search_cli_rejects_bad_arguments(same_session, tenant_a, args):
    with pytest.raises(SystemExit):
        search_cli.main(["--tenant", str(tenant_a.tenant_id), "--query", "refund", *args])


def test_search_cli_accepts_timezone_aware_datetime(same_session, tenant_a, capsys):
    assert ingest_cli.main([]) == 0
    capsys.readouterr()
    code = search_cli.main(
        [
            "--tenant",
            str(tenant_a.tenant_id),
            "--query",
            "refund",
            "--as-of",
            "2026-07-01T08:00:00+10:00",
        ]
    )
    assert code == 0 and json.loads(capsys.readouterr().out)["as_of"] == "2026-06-30"


def test_search_cli_unknown_tenant(same_session, capsys):
    assert search_cli.main(["--tenant", str(uuid.uuid4()), "--query", "refund"]) == 1
    last = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert last["error"]["code"] == "tenant_not_found"
