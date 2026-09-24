"""Policy loader: parsing, validation, path safety, duplicates and version ranges. No DB."""

import os
from datetime import date
from pathlib import Path

import pytest

from app.knowledge.sources import (
    DEFAULT_POLICY_DIR,
    PolicySourceError,
    compute_immutable_content_hash,
    discover_policy_files,
    load_policy_file,
    load_policy_sources,
    normalise,
    parse_policy,
)

TENANTS = {"northstar-commerce", "bluepeak-retail"}


def doc(
    *,
    tenant="northstar-commerce",
    key="refund-policy",
    version="1",
    effective_from="2026-01-01",
    effective_to="",
    title="Refund Policy",
    body="# Refund Policy\n\n## Timing\n\nRefunds within 5 days.\n",
    extra="",
) -> str:
    return (
        f"---\ntenant: {tenant}\ndocument_key: {key}\ntitle: {title}\ndocument_type: policy\n"
        f"version: {version}\neffective_from: {effective_from}\neffective_to: {effective_to}\n"
        f"{extra}---\n{body}"
    )


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- the real corpus -----------------------------------------------------------------------
def test_real_corpus_loads_and_is_fully_synthetic():
    sources = load_policy_sources(DEFAULT_POLICY_DIR, TENANTS)
    assert len(sources) == 12
    by_tenant = {
        t: {s.metadata.document_key for s in sources if s.metadata.tenant == t} for t in TENANTS
    }
    expected = {
        "shipping-policy",
        "delayed-shipment-compensation",
        "refund-policy",
        "returns-policy",
        "order-cancellation",
    }
    assert by_tenant["northstar-commerce"] == by_tenant["bluepeak-retail"] == expected
    for s in sources:
        assert "Fictional policy for the CommerceOps AI demo" in s.body
        for real in ("Brandhub", "brandhub", "Metis"):
            assert real not in s.body
    versioned = {
        (s.metadata.tenant, s.metadata.document_key) for s in sources if s.metadata.version > 1
    }
    assert versioned == {
        ("northstar-commerce", "refund-policy"),
        ("bluepeak-retail", "delayed-shipment-compensation"),
    }


def test_tenants_have_different_rules_for_the_same_policy():
    sources = {s.identity: s.body for s in load_policy_sources(DEFAULT_POLICY_DIR, TENANTS)}
    ns = sources[("northstar-commerce", "returns-policy", 1)]
    bp = sources[("bluepeak-retail", "returns-policy", 1)]
    assert "within 30 days" in ns and "within 14 days" in bp
    assert ns != bp


# --- parsing / validation -------------------------------------------------------------------
def test_valid_document():
    s = parse_policy(doc(effective_to="2026-07-01"), "northstar-commerce/refund-policy.v1.md")
    m = s.metadata
    assert (m.tenant, m.document_key, m.version) == ("northstar-commerce", "refund-policy", 1)
    assert (m.effective_from, m.effective_to) == (date(2026, 1, 1), date(2026, 7, 1))
    assert s.body.startswith("# Refund Policy") and len(s.immutable_content_hash) == 64


@pytest.mark.parametrize(
    "text",
    [
        "# no front matter\n",
        "---\ntenant: northstar-commerce\n# never closed\n",
        "---\n: : :\n---\nbody\n",  # not YAML
        "---\n- a\n- b\n---\nbody\n",  # not a mapping
        "---\n!!python/object/apply:os.system ['echo pwned']\n---\nbody\n",  # unsafe tag
    ],
)
def test_malformed_front_matter_is_rejected(text):
    with pytest.raises(PolicySourceError):
        parse_policy(text, "x/y.v1.md")


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"title": ""}, "title"),
        ({"version": "0"}, "version"),
        ({"version": "-1"}, "version"),
        ({"version": "'2'"}, "version"),  # a string, not an integer
        ({"version": "1.5"}, "version"),
        ({"effective_from": "2026-13-01"}, "YAML"),  # impossible date literal
        ({"effective_from": "'soon'"}, "effective_from"),
        ({"effective_to": "2025-12-31"}, "metadata"),  # before effective_from
        ({"effective_to": "2026-01-01"}, "metadata"),  # empty range
        ({"key": "Refund_Policy"}, "document_key"),
        ({"tenant": "../etc"}, "tenant"),
        ({"extra": "source_path: /etc/passwd\n"}, "source_path"),  # unknown key, no paths
    ],
)
def test_invalid_metadata_is_rejected(kwargs, field):
    with pytest.raises(PolicySourceError) as exc:
        parse_policy(doc(**kwargs), "northstar-commerce/refund-policy.v1.md")
    assert field in exc.value.reason


def test_missing_required_metadata():
    text = "---\ntenant: northstar-commerce\ndocument_key: refund-policy\n---\nbody\n"
    with pytest.raises(PolicySourceError) as exc:
        parse_policy(text, "x.md")
    for f in ("title", "version", "effective_from", "document_type"):
        assert f in exc.value.reason


def test_empty_body_is_rejected():
    with pytest.raises(PolicySourceError):
        parse_policy(doc(body="\n   \n"), "x.md")


def test_hash_is_stable_across_line_endings_but_not_content():
    lf = doc()
    crlf = lf.replace("\n", "\r\n")
    assert normalise(crlf) == lf
    assert (
        parse_policy(crlf, "a").immutable_content_hash
        == parse_policy(lf, "a").immutable_content_hash
    )
    changed = parse_policy(doc(body="# Refund Policy\n\nRefunds within 6 days.\n"), "a")
    assert changed.immutable_content_hash != parse_policy(lf, "a").immutable_content_hash
    parsed = parse_policy(lf, "a")
    assert (
        compute_immutable_content_hash(parsed.metadata, parsed.body)
        == parsed.immutable_content_hash
    )


def test_effective_to_is_not_part_of_the_immutable_hash_but_everything_else_is():
    base = parse_policy(doc(), "a").immutable_content_hash
    assert parse_policy(doc(effective_to="2026-07-01"), "a").immutable_content_hash == base
    for change in (
        {"title": "Refunds"},
        {"effective_from": "2026-01-02"},
        {"version": "2"},
        {"tenant": "bluepeak-retail"},
        {"key": "refund-policy-x"},
    ):
        assert parse_policy(doc(**change), "a").immutable_content_hash != base


# --- files, tenants and paths ----------------------------------------------------------------
def test_unknown_tenant_is_rejected(tmp_path):
    p = write(tmp_path, "acme-shop/refund-policy.v1.md", doc(tenant="acme-shop"))
    with pytest.raises(PolicySourceError, match="unknown tenant"):
        load_policy_file(tmp_path, p, TENANTS)


def test_tenant_must_match_its_directory(tmp_path):
    p = write(tmp_path, "bluepeak-retail/refund-policy.v1.md", doc(tenant="northstar-commerce"))
    with pytest.raises(PolicySourceError, match="directory"):
        load_policy_file(tmp_path, p, TENANTS)


@pytest.mark.parametrize("name", ["refund-policy.v2.md", "returns-policy.v1.md", "refund.md"])
def test_file_name_must_match_key_and_version(tmp_path, name):
    p = write(tmp_path, f"northstar-commerce/{name}", doc())
    with pytest.raises(PolicySourceError):
        load_policy_file(tmp_path, p, TENANTS)


def test_files_outside_the_policy_directory_are_refused(tmp_path):
    root = tmp_path / "policies"
    root.mkdir()
    outside = write(tmp_path, "northstar-commerce/refund-policy.v1.md", doc())
    with pytest.raises(PolicySourceError, match="outside"):
        load_policy_file(root, outside, TENANTS)
    traversal = (
        root / "northstar-commerce" / ".." / ".." / "northstar-commerce" / ("refund-policy.v1.md")
    )
    with pytest.raises(PolicySourceError, match="outside"):
        load_policy_file(root, traversal, TENANTS)


def test_symlink_escaping_the_directory_is_refused(tmp_path):
    root = tmp_path / "policies"
    (root / "northstar-commerce").mkdir(parents=True)
    secret = write(tmp_path, "secret.md", doc())
    os.symlink(secret, root / "northstar-commerce" / "refund-policy.v1.md")
    with pytest.raises(PolicySourceError):
        load_policy_sources(root, TENANTS)


def test_symlink_inside_the_directory_is_refused(tmp_path):
    real = write(tmp_path, "northstar-commerce/refund-policy.v1.md", doc())
    os.symlink(real, tmp_path / "northstar-commerce" / "refund-policy.v2.md")
    with pytest.raises(PolicySourceError):
        load_policy_sources(tmp_path, TENANTS)


def test_discovery_reads_only_tenant_markdown_files(tmp_path):
    write(tmp_path, "README.md", "# not a policy")
    write(tmp_path, "northstar-commerce/refund-policy.v1.md", doc())
    write(tmp_path, "northstar-commerce/notes.txt", "ignored")
    write(tmp_path, "northstar-commerce/deep/refund-policy.v1.md", doc())
    files = discover_policy_files(tmp_path)
    assert [f.relative_to(tmp_path).as_posix() for f in files] == [
        "northstar-commerce/refund-policy.v1.md"
    ]


def test_missing_policy_directory(tmp_path):
    with pytest.raises(PolicySourceError):
        discover_policy_files(tmp_path / "nope")


def test_invalid_utf8_is_rejected(tmp_path):
    p = tmp_path / "northstar-commerce" / "refund-policy.v1.md"
    p.parent.mkdir()
    p.write_bytes(b"---\n\xff\xfe\n---\n")
    with pytest.raises(PolicySourceError, match="UTF-8"):
        load_policy_file(tmp_path, p, TENANTS)


# --- duplicates and version ranges -----------------------------------------------------------
def test_duplicate_identity_is_rejected(tmp_path):
    a = write(tmp_path, "northstar-commerce/refund-policy.v1.md", doc())
    with pytest.raises(PolicySourceError, match="duplicate"):
        load_policy_sources(tmp_path, TENANTS, paths=[a, a])


def test_same_key_and_version_in_two_tenants_is_allowed(tmp_path):
    write(tmp_path, "northstar-commerce/refund-policy.v1.md", doc())
    write(tmp_path, "bluepeak-retail/refund-policy.v1.md", doc(tenant="bluepeak-retail"))
    assert len(load_policy_sources(tmp_path, TENANTS)) == 2


@pytest.mark.parametrize(
    ("v1", "v2", "ok"),
    [
        (("2026-01-01", "2026-07-01"), ("2026-07-01", ""), True),  # adjacent, half-open
        (("2026-01-01", "2026-03-01"), ("2026-07-01", ""), True),  # gap is allowed
        (("2026-01-01", "2026-07-02"), ("2026-07-01", ""), False),  # one-day overlap
        (("2026-01-01", ""), ("2026-07-01", ""), False),  # v1 still open
        (("2026-07-01", ""), ("2026-01-01", "2026-07-01"), False),  # v2 starts earlier
        (("2026-01-01", "2026-07-01"), ("2026-01-01", ""), False),  # same start
    ],
)
def test_version_ranges_must_not_overlap(tmp_path, v1, v2, ok):
    write(
        tmp_path,
        "northstar-commerce/refund-policy.v1.md",
        doc(effective_from=v1[0], effective_to=v1[1]),
    )
    write(
        tmp_path,
        "northstar-commerce/refund-policy.v2.md",
        doc(version="2", effective_from=v2[0], effective_to=v2[1]),
    )
    if ok:
        assert len(load_policy_sources(tmp_path, TENANTS)) == 2
    else:
        with pytest.raises(PolicySourceError):
            load_policy_sources(tmp_path, TENANTS)
