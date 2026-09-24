"""Deterministic final-answer grounding for the LangGraph RAG path.

Citations are the canonical tenant-relative strings ``policy://<key>/v<n>#chunk-<i>``. Every
``policy://`` token in the answer is extracted (up to whitespace or bracket/quote/separator
characters; trailing sentence punctuation dropped) and checked against the CURRENT-RUN
source catalog - the chunks the RETRIEVE node returned during this user invocation.

Rules (first failing rule wins):

1. retrieval status ``invalid`` (attempted, never succeeded) -> ``retrieval_required``
2. a cited URI is not in the current catalog:
   - it appeared in an earlier turn of this thread -> ``stale_citation``
   - otherwise (fabricated chunk, other version, other tenant, malformed) ->
     ``citation_not_retrieved``
3. status ``success`` and no current citation -> ``citation_required``
4. status ``no_results`` or ``none``: the catalog is empty, so any citation already failed
   rule 2; an answer without citations is accepted.

Nothing is repaired: a failing answer is never returned. Citations prove WHICH retrieved
chunks the answer points at; they do not prove each sentence is a correct reading of them.
"""

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Literal

RetrievalStatus = Literal["none", "invalid", "no_results", "success", "error"]
_TOKEN = re.compile(r"policy://[^\s\[\]()<>{}\"'`,;|]+")
_TRAILING = ".:!?"


def extract_citations(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _TOKEN.finditer(text or ""):
        token = match.group(0).rstrip(_TRAILING)
        if token:
            seen.setdefault(token, None)
    return list(seen)


@dataclass(frozen=True)
class GroundingOutcome:
    ok: bool
    detail: str | None = None
    citations: list[str] = field(default_factory=list)


def check_grounding(
    answer: str,
    *,
    status: RetrievalStatus,
    current: Mapping[str, object],
    earlier: Collection[str],
) -> GroundingOutcome:
    if status == "invalid":
        return GroundingOutcome(False, "retrieval_required")
    cited = extract_citations(answer)
    for citation in cited:
        if citation not in current:
            detail = "stale_citation" if citation in earlier else "citation_not_retrieved"
            return GroundingOutcome(False, detail)
    if status == "success" and not cited:
        return GroundingOutcome(False, "citation_required")
    return GroundingOutcome(True, None, cited)
