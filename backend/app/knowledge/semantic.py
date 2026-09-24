"""Semantic retriever ``semantic-pgvector-v1``: exact cosine search over pgvector rows.

Same contract and result shape as the lexical baseline::

    retrieve(query, context, *, as_of=None, limit=5) -> RetrievalResult

1. ``context`` must be a trusted ``AgentContext``; ``limit`` is 1-10 (else rejected);
   ``as_of`` follows the Step-7 effective-date rules (injected clock when omitted).
2. Query text is untrusted: trimmed and capped at 500 chars. Empty -> empty result, no
   provider call.
3. The provider's CONCRETE profile is resolved (model digest, cached per provider). If this
   tenant has no vectors in exactly that profile -> ``embedding_profile_not_materialized``;
   never a fallback to vectors from another digest, model, dimension or input version.
4. The query is embedded as ``search_query: <query>`` and validated (count/dims/finite/
   non-zero). The query vector is never stored.
5. ``KnowledgeQueries.nearest_chunks`` runs ONE statement with tenant, profile, dimension
   and effective-date eligibility, ordered by cosine distance.

Score: ``cosine_similarity = 1 - cosine_distance`` (pgvector ``<=>``), in [-1, 1]; higher is
more similar. It is a geometric similarity between two vectors of one model, NOT a
calibrated confidence or probability, and is not comparable across profiles.
"""

import logging
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from app.agent.context import AgentContext
from app.knowledge.embeddings.errors import EmbeddingProfileNotMaterializedError
from app.knowledge.embeddings.inputs import clean_query
from app.knowledge.embeddings.provider import (
    EmbeddingProvider,
    embed_query_text,
    resolve_profile,
)
from app.knowledge.limits import DEFAULT_LIMIT, validate_limit
from app.knowledge.retrieval import RetrievalResult, RetrievedChunk
from app.knowledge.temporal import effective_date
from app.services.knowledge import KnowledgeQueries

logger = logging.getLogger("app.knowledge.retrieval")


class SemanticKnowledgeRetriever:
    name = "semantic-pgvector-v1"

    def __init__(
        self,
        session_scope: Callable[[], AbstractContextManager[Session]],
        provider: EmbeddingProvider,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session_scope = session_scope
        self._provider = provider
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
        text = clean_query(query)
        if not text:
            result = RetrievalResult(
                retriever=self.name, as_of=day, score_type="cosine_similarity", results=[]
            )
            self._log(context, query, result, started, None)
            return result

        profile = resolve_profile(self._provider)
        with self._session_scope() as session:
            queries = KnowledgeQueries(session, context.tenant)
            if not queries.has_embeddings(profile):
                self._log(context, query, None, started, profile.key, "profile_not_materialized")
                raise EmbeddingProfileNotMaterializedError()
            vector = embed_query_text(self._provider, text)
            rows = queries.nearest_chunks(vector, profile, day, limit)

        results = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                citation=c.citation,
                document_key=c.document_key,
                title=c.title,
                version=c.version,
                section=c.section,
                chunk_index=c.chunk_index,
                effective_from=c.effective_from,
                effective_to=c.effective_to,
                content=c.content,
                score=round(similarity, 6),
                rank=i,
            )
            for i, (c, similarity) in enumerate(rows, start=1)
        ]
        result = RetrievalResult(
            retriever=self.name,
            as_of=day,
            score_type="cosine_similarity",
            embedding_profile=profile.key,
            results=results,
        )
        self._log(context, query, result, started, profile.key)
        return result

    def _log(
        self,
        context: AgentContext,
        query: str,
        result: RetrievalResult | None,
        started: float,
        profile_key: str | None,
        outcome: str = "ok",
    ) -> None:
        # Profile, counts, citations and timing only: never query text, vectors or content.
        fields = {
            "retriever": self.name,
            "tenant_id": str(context.tenant_id),
            "profile": profile_key,
            "query_chars": len(query) if isinstance(query, str) else 0,
            "result_count": len(result.results) if result else 0,
            "citations": [r.citation for r in result.results] if result else [],
            "outcome": outcome,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if result is not None:
            fields["as_of"] = result.as_of.isoformat()
        if context.request_id:
            fields["request_id"] = context.request_id
        logger.log(
            logging.INFO if outcome == "ok" else logging.WARNING,
            "knowledge retrieval",
            extra=fields,
        )
