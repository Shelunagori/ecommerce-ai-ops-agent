"""The committed semantic baseline was measured against exactly the current policy corpus."""

import json

from app.knowledge.chunking import CHUNKER, ChunkingConfig
from app.knowledge.evaluation import DEFAULT_CASES, corpus_fingerprint
from app.knowledge.ingest import ingest_policies
from app.knowledge.sources import DEFAULT_POLICY_DIR

SEMANTIC = DEFAULT_CASES.parent / "semantic_baseline_v1.json"


def test_corpus_fingerprint_matches_the_committed_semantic_baseline(db_session):
    ingest_policies(db_session, DEFAULT_POLICY_DIR, ChunkingConfig())
    snap = json.loads(SEMANTIC.read_text(encoding="utf-8"))
    assert snap["chunker"] == CHUNKER
    assert snap["corpus_sha256"] == corpus_fingerprint(db_session)
