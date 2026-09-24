"""Read models for the knowledge domain (no tenant ids, no filesystem paths)."""

import uuid
from datetime import date

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)


class KnowledgeDocumentRead(_Frozen):
    document_key: str
    title: str
    document_type: str
    version: int
    effective_from: date
    effective_to: date | None
    chunk_count: int


class KnowledgeChunkRead(_Frozen):
    chunk_id: uuid.UUID
    citation: str
    document_key: str
    title: str
    version: int
    section: str
    chunk_index: int
    effective_from: date
    effective_to: date | None
    content: str
