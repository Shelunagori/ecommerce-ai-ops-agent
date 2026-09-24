"""policy-section-v1 chunker: structure, determinism, size boundaries, identity. No DB."""

import uuid

import pytest

from app.core.config import Settings
from app.knowledge.chunking import CHUNKER, ChunkingConfig, chunk_policy
from app.knowledge.ingest import chunk_id_for, document_id_for
from app.knowledge.sources import DEFAULT_POLICY_DIR, load_policy_sources

C = ChunkingConfig()
BODY = """# Refund Policy

> Fictional.

## Eligibility

- Item unused
- Within **30 days**

| Tier | Days |
| --- | --- |
| Standard | 30 |

## Timing

### Card payments

Refunds in 5 business days. Café ✓ — ünïcödé.

```text
## not a heading inside a code fence
```
"""


def test_sections_carry_heading_paths_in_order():
    chunks = chunk_policy(BODY, "Refund Policy", C)
    assert [(c.index, c.section) for c in chunks] == [
        (0, "Refund Policy"),
        (1, "Refund Policy > Eligibility"),
        (2, "Refund Policy > Timing > Card payments"),
    ]
    assert "## not a heading" in chunks[2].content  # fenced code is not a heading


def test_lists_tables_and_unicode_are_kept_verbatim():
    chunks = chunk_policy(BODY, "Refund Policy", C)
    assert "- Item unused\n- Within **30 days**\n\n| Tier | Days |" in chunks[1].content
    assert "Café ✓ — ünïcödé." in chunks[2].content
    assert chunks[2].char_count == len(chunks[2].content)  # code points, like char_length()


def test_no_empty_chunks_and_headings_without_body_are_skipped():
    body = "# Title\n\n## Empty\n\n## Filled\n\nText.\n\n## Also empty\n"
    chunks = chunk_policy(body, "Title", C)
    assert [c.section for c in chunks] == ["Title > Filled"]


def test_section_path_is_the_heading_stack_not_a_title_prefix():
    chunks = chunk_policy("## A\n\nx\n\n### B\n\ny\n\n## C\n\nz\n", "Doc", C)
    assert [c.section for c in chunks] == ["A", "A > B", "C"]
    assert all(c.content.strip() for c in chunks)


def test_text_before_any_heading_uses_the_title():
    assert chunk_policy("Just text.", "Doc Title", C)[0].section == "Doc Title"


def test_empty_document_is_an_error():
    with pytest.raises(ValueError):
        chunk_policy("# Only a heading\n", "T", C)


def test_deterministic_output_and_ids():
    a = chunk_policy(BODY, "Refund Policy", C)
    b = chunk_policy(BODY, "Refund Policy", C)
    assert a == b
    doc = document_id_for(uuid.UUID(int=7), "refund-policy", 1)
    assert [chunk_id_for(doc, c.index, c.content_hash) for c in a] == [
        chunk_id_for(doc, c.index, c.content_hash) for c in b
    ]
    assert doc == document_id_for(uuid.UUID(int=7), "refund-policy", 1)
    assert doc != document_id_for(uuid.UUID(int=8), "refund-policy", 1)  # tenant-scoped
    assert doc != document_id_for(uuid.UUID(int=7), "refund-policy", 2)


def test_changed_content_changes_only_the_affected_chunk_identity():
    changed = BODY.replace("5 business days", "6 business days")
    a, b = chunk_policy(BODY, "Refund Policy", C), chunk_policy(changed, "Refund Policy", C)
    assert [x.content_hash == y.content_hash for x, y in zip(a, b, strict=True)] == [
        True,
        True,
        False,
    ]
    doc = document_id_for(uuid.UUID(int=7), "refund-policy", 1)
    assert chunk_id_for(doc, 2, a[2].content_hash) != chunk_id_for(doc, 2, b[2].content_hash)


def test_section_at_exactly_max_chars_is_one_chunk():
    cfg = ChunkingConfig(max_chars=200)
    text = "x" * 200
    [chunk] = chunk_policy(f"## S\n\n{text}\n", "T", cfg)
    assert chunk.char_count == 200
    parts = chunk_policy(f"## S\n\n{text}y\n", "T", cfg)
    assert len(parts) == 2 and all(p.char_count <= 200 for p in parts)


def test_oversized_section_splits_at_blocks_then_sentences():
    cfg = ChunkingConfig(max_chars=200)
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    para = " ".join(f"Sentence number {i} is here." for i in range(20))  # ~560 chars
    chunks = chunk_policy(f"## Big\n\n{table}\n\n{para}\n", "T", cfg)
    assert chunks[0].content == table  # the table block stays whole
    assert all(c.char_count <= 200 and c.section == "Big" for c in chunks)
    assert all(c.content.endswith(".") for c in chunks[1:])  # sentence boundaries
    rejoined = " ".join(c.content for c in chunks[1:])
    assert rejoined == para  # nothing lost or duplicated without overlap


def test_unbreakable_text_is_hard_cut():
    cfg = ChunkingConfig(max_chars=200)
    chunks = chunk_policy("## S\n\n" + "y" * 450 + "\n", "T", cfg)
    assert [c.char_count for c in chunks] == [200, 200, 50]


def test_overlap_only_within_a_split_section_and_never_exceeds_max():
    cfg = ChunkingConfig(max_chars=200, overlap_chars=40)
    para = " ".join(f"Clause {i} applies to every order." for i in range(15))
    body = f"## First\n\n{para}\n\n## Second\n\nShort section.\n"
    chunks = chunk_policy(body, "T", cfg)
    first = [c for c in chunks if c.section == "First"]
    assert len(first) > 1 and all(c.char_count <= 200 for c in chunks)
    for prev, cur in zip(first, first[1:], strict=False):
        head = cur.content.split("\n", 1)[0]
        assert head and head in prev.content  # tail of the previous piece is repeated
    [second] = [c for c in chunks if c.section == "Second"]
    assert second.content == "Short section."  # no overlap across sections


def test_chunking_hash_records_chunker_and_settings():
    base = ChunkingConfig().chunking_hash
    assert base == ChunkingConfig(1200, 0).chunking_hash
    assert base != ChunkingConfig(1000, 0).chunking_hash
    assert base != ChunkingConfig(1200, 50).chunking_hash
    assert CHUNKER == "policy-section-v1"


@pytest.mark.parametrize(("max_chars", "overlap"), [(199, 0), (4001, 0), (400, 301), (400, 200)])
def test_config_bounds(max_chars, overlap):
    with pytest.raises(ValueError):
        ChunkingConfig(max_chars, overlap)


def test_config_from_trusted_settings():
    s = Settings(knowledge_chunk_max_chars=800, knowledge_chunk_overlap_chars=40)
    assert ChunkingConfig.from_settings(s) == ChunkingConfig(800, 40)
    assert ChunkingConfig.from_settings(Settings()) == ChunkingConfig(1200, 0)


def test_real_corpus_chunks_one_per_section_under_defaults():
    sources = load_policy_sources(DEFAULT_POLICY_DIR, {"northstar-commerce", "bluepeak-retail"})
    total = 0
    for s in sources:
        chunks = chunk_policy(s.body, s.metadata.title, C)
        headings = [line for line in s.body.splitlines() if line.startswith("## ")]
        assert len(chunks) == len(headings) + 1  # disclaimer + one chunk per section
        assert all(0 < c.char_count <= C.max_chars for c in chunks)
        assert "tenant" not in " ".join(c.content.lower() for c in chunks)
        total += len(chunks)
    assert total == 51
