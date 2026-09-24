"""``policy-embedding-input-v1``: the exact text that gets embedded.

nomic-embed-text models are trained with task prefixes, so the prefixes live HERE, not in
callers:

    document:  "search_document: Title: <title>\\nSection: <section path>\\n\\n<chunk content>"
    query:     "search_query: <query>"

Never included: tenant ids, database ids, file paths or citations. Any change to this
format requires a new ``INPUT_VERSION`` (it is part of the embedding profile), otherwise
stored vectors would silently describe different text (``input_hash`` detects it).

Inputs are never truncated here, in the provider, or by Ollama (the provider sends
``truncate=False``): the model has a finite context (nomic-embed-text-v2-moe: 512 tokens),
and silently cutting text would make the recorded input hash lie about what was embedded.
Policy chunks are bounded by the chunker instead; an over-long input fails loudly.
"""

import hashlib

INPUT_VERSION = "policy-embedding-input-v1"
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
MAX_QUERY_CHARS = 500


def document_input(title: str, section: str, content: str) -> str:
    return f"{DOCUMENT_PREFIX}Title: {title}\nSection: {section}\n\n{content}"


def query_input(query: str) -> str:
    return f"{QUERY_PREFIX}{query}"


def clean_query(query: str) -> str:
    """Untrusted query text: must be a string; trimmed and capped at 500 chars."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    return query[:MAX_QUERY_CHARS].strip()


def input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
