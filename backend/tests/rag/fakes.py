"""Deterministic stand-ins for the policy retriever (graph-level RAG tests, no database).

``FakeRetriever`` records every call (query, trusted tenant, as_of, limit) and returns
scripted ``RetrievalResult`` objects, or raises scripted exceptions. Real semantic retrieval
against PostgreSQL + pgvector is covered in ``tests/db/test_rag_graph_db.py``.
"""

import itertools
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from app.agent.rag.capability import SEARCH_POLICY_KNOWLEDGE
from app.knowledge.retrieval import RetrievalResult, RetrievedChunk

SEPT = date(2026, 9, 1)
JUNE = date(2026, 6, 10)
_ids = itertools.count(1)


def chunk(
    key: str = "delayed-shipment-compensation",
    version: int = 2,
    index: int = 2,
    *,
    title: str = "Delayed Shipment Compensation Policy",
    section: str = "Delayed Shipment Compensation Policy > Compensation",
    content: str = "Customers receive store credit worth 15% of the order total, capped at 25 EUR.",
    effective_from: date = date(2026, 8, 15),
    effective_to: date | None = None,
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        citation=f"policy://{key}/v{version}#chunk-{index}",
        document_key=key,
        title=title,
        version=version,
        section=section,
        chunk_index=index,
        effective_from=effective_from,
        effective_to=effective_to,
        content=content,
        score=0.9 - rank / 100,
        rank=rank,
    )


def result(*chunks: RetrievedChunk, as_of: date = SEPT) -> RetrievalResult:
    return RetrievalResult(
        retriever="semantic-pgvector-v1",
        as_of=as_of,
        score_type="cosine_similarity",
        embedding_profile="ollama/fake:latest@" + "a" * 64 + "/768/policy-embedding-input-v1",
        results=list(chunks),
    )


@dataclass
class RetrieverCall:
    query: str
    tenant_id: uuid.UUID
    request_id: str | None
    as_of: Any
    limit: int


@dataclass
class FakeRetriever:
    """Plays ``script`` (RetrievalResult | exception) in order; the default is one chunk."""

    script: list[Any] = field(default_factory=list)
    calls: list[RetrieverCall] = field(default_factory=list)
    clock_day: date = SEPT

    def retrieve(self, query, context, *, as_of=None, limit=5):
        self.calls.append(RetrieverCall(query, context.tenant_id, context.request_id, as_of, limit))
        step = self.script.pop(0) if self.script else result(chunk())
        if isinstance(step, BaseException):
            raise step
        return step


def search(query: str = "delayed shipment compensation", call_id: str | None = None, **extra):
    args: dict[str, Any] = {"query": query, **extra}
    return {
        "name": SEARCH_POLICY_KNOWLEDGE,
        "args": args,
        "id": call_id or f"s_{next(_ids)}",
        "type": "tool_call",
    }


def policy_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage) and m.name == SEARCH_POLICY_KNOWLEDGE]
