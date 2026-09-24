"""The single authoritative definition of knowledge-retrieval result limits.

Deliberately dependency-free so the lexical retriever, the semantic retriever, the
tenant-scoped service/query boundary and the CLIs can all import it without cycles.
"""

DEFAULT_LIMIT = 5
MAX_LIMIT = 10  # hard maximum for every retriever and every vector query


def validate_limit(limit: int) -> int:
    """Return ``limit`` if it is an int in 1..MAX_LIMIT, else raise ``ValueError``."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer between 1 and {MAX_LIMIT}")
    return limit
