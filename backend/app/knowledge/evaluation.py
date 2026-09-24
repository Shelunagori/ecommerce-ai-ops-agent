"""Retrieval evaluation over ``data/eval/policy_retrieval_cases.yaml`` (retriever-agnostic).

Metrics per tenant and overall: document hit@1 / hit@3 (expected document_key AND version
among the top k) and chunk hit@1 / hit@3 (exact expected citation among the top k). Any
``KnowledgeRetriever`` can be scored against the same cases, so Step 8 semantic retrieval
is compared with the Step 7 lexical baseline on identical inputs.
"""

import hashlib
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.context import AgentContext
from app.knowledge.citations import parse_citation
from app.knowledge.retrieval import KnowledgeRetriever
from app.models import KnowledgeDocument

DEFAULT_CASES = (
    Path(__file__).resolve().parents[2] / "data" / "eval" / "policy_retrieval_cases.yaml"
)


class RetrievalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tenant: str
    as_of: date
    query: str = Field(min_length=1, max_length=500)
    expected: str


class _CaseFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    cases: list[RetrievalCase]


@dataclass(frozen=True)
class CaseOutcome:
    case: RetrievalCase
    citations: tuple[str, ...]  # top-k retrieved, in rank order

    def document_rank(self) -> int | None:
        want = parse_citation(self.case.expected)
        for i, c in enumerate(self.citations, start=1):
            got = parse_citation(c)
            if (got.document_key, got.version) == (want.document_key, want.version):
                return i
        return None

    def chunk_rank(self) -> int | None:
        return next((i for i, c in enumerate(self.citations, 1) if c == self.case.expected), None)


def load_cases(path: Path = DEFAULT_CASES) -> list[RetrievalCase]:
    data = _CaseFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    ids = [c.id for c in data.cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case id")
    for c in data.cases:
        parse_citation(c.expected)
    return data.cases


def run_cases(
    retriever: KnowledgeRetriever,
    cases: list[RetrievalCase],
    tenant_ids: Mapping[str, uuid.UUID],
    *,
    k: int = 3,
) -> list[CaseOutcome]:
    outcomes = []
    for case in cases:
        ctx = AgentContext(tenant_ids[case.tenant], f"eval-{case.id}"[:64])
        result = retriever.retrieve(case.query, ctx, as_of=case.as_of, limit=k)
        outcomes.append(CaseOutcome(case, tuple(r.citation for r in result.results)))
    return outcomes


def metrics(outcomes: list[CaseOutcome]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[CaseOutcome]] = {"all": list(outcomes)}
    for o in outcomes:
        groups.setdefault(o.case.tenant, []).append(o)

    def rate(items: list[CaseOutcome], fn: Callable[[CaseOutcome], int | None], k: int) -> float:
        hits = sum(1 for o in items if (r := fn(o)) is not None and r <= k)
        return round(hits / len(items), 3)

    return {
        name: {
            "cases": len(items),
            "document_hit@1": rate(items, CaseOutcome.document_rank, 1),
            "document_hit@3": rate(items, CaseOutcome.document_rank, 3),
            "chunk_hit@1": rate(items, CaseOutcome.chunk_rank, 1),
            "chunk_hit@3": rate(items, CaseOutcome.chunk_rank, 3),
        }
        for name, items in sorted(groups.items())
    }


def per_case(outcomes: list[CaseOutcome]) -> dict[str, dict[str, int | None]]:
    return {
        o.case.id: {"document_rank": o.document_rank(), "chunk_rank": o.chunk_rank()}
        for o in outcomes
    }


def cases_fingerprint(path: Path = DEFAULT_CASES) -> str:
    """sha256 of the evaluation-cases file (line endings normalised)."""
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def corpus_fingerprint(session: Session) -> str:
    """sha256 over every stored document identity + immutable hash + chunking hash (all
    tenants, sorted): identifies the exact corpus an evaluation ran against."""
    rows = session.execute(
        select(
            KnowledgeDocument.tenant_id,
            KnowledgeDocument.document_key,
            KnowledgeDocument.version,
            KnowledgeDocument.immutable_content_hash,
            KnowledgeDocument.chunking_hash,
        ).order_by(
            KnowledgeDocument.tenant_id, KnowledgeDocument.document_key, KnowledgeDocument.version
        )
    ).all()
    payload = "\n".join("|".join(str(v) for v in r) for r in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def comparison(
    baseline: dict[str, dict[str, int | None]], candidate: dict[str, dict[str, int | None]]
) -> list[dict[str, object]]:
    """Per-case chunk-rank comparison (lower rank is better; None = not in top k)."""

    def score(rank: int | None) -> int:
        return rank if rank is not None else 99

    rows = []
    for case_id in baseline:
        b, c = baseline[case_id]["chunk_rank"], candidate[case_id]["chunk_rank"]
        outcome = "tie" if score(b) == score(c) else ("win" if score(c) < score(b) else "loss")
        rows.append(
            {
                "case": case_id,
                "lexical_chunk_rank": b,
                "semantic_chunk_rank": c,
                "lexical_document_rank": baseline[case_id]["document_rank"],
                "semantic_document_rank": candidate[case_id]["document_rank"],
                "semantic_vs_lexical": outcome,
            }
        )
    return rows
