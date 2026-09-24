"""Retrieval contract and the Step-7 deterministic lexical baseline (no LLM, no embeddings).

``KnowledgeRetriever`` is the provider-neutral interface later semantic retrievers will
implement. ``LexicalPolicyRetriever`` ("lexical-pg-fts-v1") is a transparent baseline:

1. Query text is untrusted. It is cut to 500 chars, lowercased, and reduced to at most 32
   distinct ``[a-z0-9]+`` terms (the word "or" dropped); all punctuation and search
   operators are discarded. No terms -> empty result, no database call.
2. The surviving terms are joined as ``"t1 or t2 or ..."`` and passed as a bound parameter
   to ``websearch_to_tsquery('english', ...)`` (English stemming + stop words). Any-term
   (OR) matching, so natural-language questions still match.
3. Stop-word-only queries ("the", "the and of") survive step 1 but PostgreSQL's English
   dictionary removes every term, leaving a tsquery with zero nodes. A pre-check measures
   ``numnode`` of the built tsquery; zero -> empty result and the search SQL never runs
   (never a fallback to unfiltered or rank-zero results). The search SQL also keeps
   ``numnode(query) > 0`` as a second guard.
4. Each chunk is scored with ``ts_rank_cd(vector, query, 1)`` where
   ``vector = setweight(title, 'A') || setweight(section, 'A') || setweight(content, 'B')``
   (all ``to_tsvector('english', ...)``): title and heading words weigh more than body
   words; normalisation 1 divides by 1 + log(length).
5. Tenant and effective-version filters are in the SAME SQL statement as the ranking:
   ``chunk.tenant_id = :tenant AND effective_from <= :as_of < effective_to``.
6. Order: score DESC, document_key, version DESC, chunk_index (total, deterministic).
"""

import logging
import re
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, date, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.agent.context import AgentContext
from app.knowledge.citations import citation_for
from app.knowledge.temporal import effective_date

logger = logging.getLogger("app.knowledge.retrieval")

DEFAULT_LIMIT = 5
MAX_LIMIT = 10
MAX_QUERY_CHARS = 500
MAX_QUERY_TERMS = 32
_TERM = re.compile(r"[a-z0-9]+")
_RESERVED_TERMS = frozenset({"or"})  # websearch_to_tsquery operator word


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: uuid.UUID
    citation: str  # tenant-relative: policy://<key>/v<version>#chunk-<n>
    document_key: str
    title: str
    version: int
    section: str
    chunk_index: int
    effective_from: date
    effective_to: date | None
    content: str
    score: float
    rank: int


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    retriever: str
    as_of: date
    term_count: int  # trusted [a-z0-9]+ terms extracted from the query text
    lexeme_count: int  # meaningful lexemes left after PostgreSQL stemming / stop words
    results: list[RetrievedChunk]


class KnowledgeRetriever(Protocol):
    name: str

    def retrieve(
        self,
        query: str,
        context: AgentContext,
        *,
        as_of: date | datetime | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> RetrievalResult: ...


def query_terms(query: str) -> list[str]:
    """Trusted search terms extracted from untrusted text (order-preserving, deduplicated)."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    terms: list[str] = []
    for term in _TERM.findall(query[:MAX_QUERY_CHARS].lower()):
        if term not in _RESERVED_TERMS and term not in terms:
            terms.append(term)
            if len(terms) == MAX_QUERY_TERMS:
                break
    return terms


def validate_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer between 1 and {MAX_LIMIT}")
    return limit


_LEXEME_SQL = text(
    """
    SELECT numnode(websearch_to_tsquery('english', :tsquery)) AS nodes,
           coalesce(array_length(tsvector_to_array(to_tsvector('english', :tsquery)), 1), 0)
               AS lexemes
    """
).bindparams(bindparam("tsquery"))

_SEARCH_SQL = text(
    """
    WITH q AS (SELECT websearch_to_tsquery('english', :tsquery) AS query)
    SELECT c.id, c.chunk_index, c.section, c.content,
           d.document_key, d.title, d.version, d.effective_from, d.effective_to,
           ts_rank_cd(
               setweight(to_tsvector('english', d.title), 'A')
               || setweight(to_tsvector('english', c.section), 'A')
               || setweight(to_tsvector('english', c.content), 'B'),
               q.query, 1
           ) AS score
    FROM knowledge_chunks AS c
    JOIN knowledge_documents AS d ON d.tenant_id = c.tenant_id AND d.id = c.document_id
    CROSS JOIN q
    WHERE c.tenant_id = :tenant_id
      AND d.tenant_id = :tenant_id
      AND d.effective_from <= :as_of
      AND (d.effective_to IS NULL OR :as_of < d.effective_to)
      AND numnode(q.query) > 0
      AND (
          setweight(to_tsvector('english', d.title), 'A')
          || setweight(to_tsvector('english', c.section), 'A')
          || setweight(to_tsvector('english', c.content), 'B')
      ) @@ q.query
    ORDER BY score DESC, d.document_key ASC, d.version DESC, c.chunk_index ASC
    LIMIT :limit
    """
).bindparams(bindparam("tsquery"), bindparam("tenant_id"), bindparam("as_of"), bindparam("limit"))


class LexicalPolicyRetriever:
    name = "lexical-pg-fts-v1"

    def __init__(
        self,
        session_scope: Callable[[], AbstractContextManager[Session]],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session_scope = session_scope
        self._clock = clock

    def retrieve(
        self,
        query: str,
        context: AgentContext,
        *,
        as_of: date | datetime | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> RetrievalResult:
        if not isinstance(context, AgentContext):
            raise TypeError("context must be a validated AgentContext")
        started = time.perf_counter()
        limit = validate_limit(limit)
        day = effective_date(as_of if as_of is not None else self._clock())
        terms = query_terms(query)
        results: list[RetrievedChunk] = []
        lexemes = 0
        rows: list = []
        if terms:
            tsquery = " or ".join(terms)
            with self._session_scope() as session:
                check = session.execute(_LEXEME_SQL, {"tsquery": tsquery}).one()
                lexemes = int(check.lexemes)
                if check.nodes > 0:  # stop-word-only -> nothing searchable -> empty result
                    rows = session.execute(
                        _SEARCH_SQL,
                        {
                            "tsquery": tsquery,
                            "tenant_id": context.tenant_id,
                            "as_of": day,
                            "limit": limit,
                        },
                    ).all()
            results = [
                RetrievedChunk(
                    chunk_id=r.id,
                    citation=citation_for(r.document_key, r.version, r.chunk_index),
                    document_key=r.document_key,
                    title=r.title,
                    version=r.version,
                    section=r.section,
                    chunk_index=r.chunk_index,
                    effective_from=r.effective_from,
                    effective_to=r.effective_to,
                    content=r.content,
                    score=round(float(r.score), 6),
                    rank=i,
                )
                for i, r in enumerate(rows, start=1)
            ]
        result = RetrievalResult(
            retriever=self.name,
            as_of=day,
            term_count=len(terms),
            lexeme_count=lexemes,
            results=results,
        )
        self._log(context, query, result, started)
        return result

    def _log(
        self, context: AgentContext, query: str, result: RetrievalResult, started: float
    ) -> None:
        # Counts and citations only: never the query text or chunk content.
        fields = {
            "retriever": self.name,
            "tenant_id": str(context.tenant_id),
            "as_of": result.as_of.isoformat(),
            "query_chars": len(query) if isinstance(query, str) else 0,
            "term_count": result.term_count,
            "lexeme_count": result.lexeme_count,
            "result_count": len(result.results),
            "citations": [r.citation for r in result.results],
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if context.request_id:
            fields["request_id"] = context.request_id
        logger.info("knowledge retrieval", extra=fields)
