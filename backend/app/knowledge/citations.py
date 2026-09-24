"""Tenant-RELATIVE citations for policy chunks: ``policy://<document_key>/v<version>#chunk-<n>``.

A citation is model-visible and carries no tenant, id or filesystem path, so it is NOT
globally unique: the same string names different chunks in different tenants. Resolving a
citation is therefore only ever done inside a trusted tenant scope
(``KnowledgeQueries.get_chunk_by_citation``); a citation alone never selects a tenant.
"""

import re
from dataclasses import dataclass

_CITATION = re.compile(
    r"^policy://(?P<key>[a-z0-9]+(?:-[a-z0-9]+)*)/v(?P<version>[1-9][0-9]{0,3})"
    r"#chunk-(?P<index>0|[1-9][0-9]{0,4})$"
)


@dataclass(frozen=True)
class CitationRef:
    document_key: str
    version: int
    chunk_index: int


def citation_for(document_key: str, version: int, chunk_index: int) -> str:
    return f"policy://{document_key}/v{version}#chunk-{chunk_index}"


def parse_citation(citation: str) -> CitationRef:
    match = _CITATION.fullmatch(citation) if isinstance(citation, str) else None
    if not match:
        raise ValueError("not a policy citation")
    return CitationRef(match["key"], int(match["version"]), int(match["index"]))
