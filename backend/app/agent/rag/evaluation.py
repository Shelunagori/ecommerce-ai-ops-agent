"""RAG / agent evaluation (Step 9): deterministic structured expectations, no LLM judge.

Cases: ``data/eval/rag_agent_cases.yaml`` (separate from the frozen Step-7/8 retrieval
benchmark). Each case runs on a fresh in-memory thread of the production LangGraph assistant
(``RAG_PROFILE``). The retriever uses a FIXED evaluation clock (the case's ``as_of``), so
committed metrics never depend on the wall clock. Only the LAST turn is scored.

Per-case outcome (structured, never prose): retrieval decision, commerce tools used,
retrieved document versions, final citations, no-result abstention, tenant isolation,
error code/detail, pass/fail. Metrics are rates over the cases where a check applies.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.assistant import AssistantError, AssistantLimits
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.checkpoint import ephemeral_checkpointer
from app.agent.graph.nodes import PolicyRetriever
from app.agent.llm import LLMProvider
from app.knowledge.retrieval import RetrievalResult

DEFAULT_RAG_CASES = Path(__file__).resolve().parents[3] / "data" / "eval" / "rag_agent_cases.yaml"
_SOURCE = r"^policy://[a-z0-9]+(-[a-z0-9]+)*/v[1-9][0-9]*$"

METRIC_NAMES = (
    "retrieval_decision_accuracy",
    "commerce_trajectory_accuracy",
    "correct_document_version_rate",
    "citation_presence_rate",
    "citation_validity_rate",
    "no_result_abstention_rate",
    "stale_citation_safety_rate",
    "tenant_isolation_pass_rate",
    "case_pass_rate",
)


class CommerceCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tool: str
    args: dict[str, str] = {}


class RagCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9-]{1,64}$")
    tenant: str
    as_of: date  # trusted evaluation clock
    turns: list[str] = Field(min_length=1, max_length=4)
    retrieval_required: bool
    model_as_of: date | None = None
    expected_sources: list[str] = []
    forbidden_sources: list[str] = []
    expected_commerce_tools: list[CommerceCall] = []
    expect_no_result: bool = False
    isolation_group: str | None = None

    @field_validator("expected_sources", "forbidden_sources")
    @classmethod
    def _sources(cls, values: list[str]) -> list[str]:
        for v in values:
            if not re.fullmatch(_SOURCE, v):
                raise ValueError(f"source must look like policy://<key>/v<n>: {v}")
        return values


class _CaseFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int
    cases: list[RagCase]


def load_rag_cases(path: Path = DEFAULT_RAG_CASES) -> list[RagCase]:
    data = _CaseFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    ids = [c.id for c in data.cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case id")
    for c in data.cases:
        if c.expect_no_result and (c.expected_sources or not c.retrieval_required):
            raise ValueError(f"{c.id}: a no-result case retrieves and expects no sources")
    return data.cases


def rag_cases_fingerprint(path: Path = DEFAULT_RAG_CASES) -> str:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def source_of(citation: str) -> str:
    """``policy://key/v2#chunk-3`` -> ``policy://key/v2``."""
    return citation.split("#", 1)[0]


@dataclass
class _Record:
    tenant_id: uuid.UUID
    as_of: date
    citations: list[str]


@dataclass
class RecordingPolicyRetriever:
    """Wraps the real retriever; records what was retrieved for whom (no query text kept)."""

    inner: PolicyRetriever
    records: list[_Record] = field(default_factory=list)

    def retrieve(self, query: str, context: AgentContext, **kw: Any) -> RetrievalResult:
        result = self.inner.retrieve(query, context, **kw)
        self.records.append(
            _Record(context.tenant_id, result.as_of, [r.citation for r in result.results])
        )
        return result


def run_rag_case(
    case: RagCase,
    provider: LLMProvider,
    retriever_for: Callable[[date], PolicyRetriever],
    tools: list[BaseTool] | tuple[BaseTool, ...],
    tenant_id: uuid.UUID,
    *,
    limits: AssistantLimits | None = None,
) -> dict[str, Any]:
    recorder = RecordingPolicyRetriever(retriever_for(case.as_of))
    assistant = CommerceGraphAssistant(
        provider,
        tools=tools,
        limits=limits or AssistantLimits(),
        retriever=recorder,
        checkpointer=ephemeral_checkpointer(),
    )
    context = AgentContext(tenant_id, f"rag-eval-{case.id}"[:64])
    thread = f"eval-{case.id}"[:64]
    result = error = None
    failed_turn = None
    for i, question in enumerate(case.turns):
        if i == len(case.turns) - 1:
            recorder.records.clear()  # score the last turn only
        try:
            result = assistant.run(question, context, thread_id=thread)
        except AssistantError as exc:
            error, failed_turn = exc, i + 1
            break
    return score_case(case, tenant_id, result, error, failed_turn, recorder.records)


def run_rag_cases(
    cases: list[RagCase],
    provider_for: Callable[[RagCase], LLMProvider],
    retriever_for: Callable[[date], PolicyRetriever],
    tools: list[BaseTool] | tuple[BaseTool, ...],
    tenants: Mapping[str, uuid.UUID],
) -> list[dict[str, Any]]:
    return [
        run_rag_case(c, provider_for(c), retriever_for, tools, tenants[c.tenant]) for c in cases
    ]


def score_case(
    case: RagCase,
    tenant_id: uuid.UUID,
    result: Any,
    error: AssistantError | None,
    failed_turn: int | None,
    records: list[_Record],
) -> dict[str, Any]:
    last_turn = len(case.turns)
    ok = error is None and result is not None
    retrievals = (result.retrievals if ok else error.retrievals if error else []) or []
    tool_calls = (result.tool_calls if ok else error.tool_calls if error else []) or []
    citations = [c.citation for c in result.citations] if ok else []
    retrieved = sorted({c for r in records for c in r.citations})
    retrieved_sources = sorted({source_of(c) for c in retrieved})
    executed = [r for r in retrievals if r.outcome in ("success", "no_results")]
    performed = bool(executed) and (failed_turn in (None, last_turn))
    forbidden = set(case.forbidden_sources)
    wants_citation = case.retrieval_required and not case.expect_no_result

    outcome: dict[str, Any] = {
        "case": case.id,
        "tenant": case.tenant,
        "turns": last_turn,
        "ok": ok,
        "error_code": error.code if error else None,
        "error_detail": error.detail if error else None,
        "failed_turn": failed_turn,
        "retrieval_performed": performed,
        "retrieval_decision_correct": performed == case.retrieval_required,
        "retrieval_as_of": sorted({r.as_of.isoformat() for r in records}),
        "commerce_tools": [c.tool for c in tool_calls],
        "commerce_trajectory_correct": [c.tool for c in tool_calls]
        == [c.tool for c in case.expected_commerce_tools],
        "retrieved_sources": retrieved_sources,
        "citations": citations,
    }
    outcome["correct_document_version"] = (
        set(case.expected_sources) <= set(retrieved_sources)
        and not (forbidden & set(retrieved_sources))
        if wants_citation
        else None
    )
    outcome["citation_present"] = bool(citations) if wants_citation else None
    # Valid: every final citation was retrieved in this turn, from an expected (never a
    # forbidden) document; answers that need no policy cite nothing.
    outcome["citations_valid"] = (
        all(c in retrieved for c in citations)
        and all(source_of(c) not in forbidden for c in citations)
        and (
            not case.expected_sources
            or all(source_of(c) in case.expected_sources for c in citations)
        )
        and (wants_citation or not citations)
        if ok
        else None
    )
    outcome["no_result_abstained"] = (
        ok and performed and not retrieved and not citations if case.expect_no_result else None
    )
    outcome["stale_citation_safe"] = (
        (ok and performed and outcome["citations_valid"] is True)
        or (error is not None and error.detail == "stale_citation")
        if last_turn > 1
        else None
    )
    outcome["tenant_isolation_ok"] = all(r.tenant_id == tenant_id for r in records) and not (
        forbidden & ({source_of(c) for c in citations} | set(retrieved_sources))
    )
    checks = [
        outcome["retrieval_decision_correct"],
        outcome["commerce_trajectory_correct"],
        outcome["correct_document_version"],
        outcome["citation_present"],
        outcome["citations_valid"],
        outcome["no_result_abstained"],
        outcome["stale_citation_safe"],
        outcome["tenant_isolation_ok"],
    ]
    outcome["passed"] = ok and all(c is not False for c in checks)
    return outcome


def _rate(outcomes: list[dict[str, Any]], key: str) -> float | None:
    values = [o[key] for o in outcomes if o[key] is not None]
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 3)


def rag_metrics(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(outcomes),
        "retrieval_decision_accuracy": _rate(outcomes, "retrieval_decision_correct"),
        "commerce_trajectory_accuracy": _rate(outcomes, "commerce_trajectory_correct"),
        "correct_document_version_rate": _rate(outcomes, "correct_document_version"),
        "citation_presence_rate": _rate(outcomes, "citation_present"),
        "citation_validity_rate": _rate(outcomes, "citations_valid"),
        "no_result_abstention_rate": _rate(outcomes, "no_result_abstained"),
        "stale_citation_safety_rate": _rate(outcomes, "stale_citation_safe"),
        "tenant_isolation_pass_rate": _rate(outcomes, "tenant_isolation_ok"),
        "case_pass_rate": _rate(outcomes, "passed"),
    }
