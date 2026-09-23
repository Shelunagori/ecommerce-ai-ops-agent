"""Normalise PostgreSQL URLs to the SQLAlchemy + psycopg 3 form."""

DRIVER_SCHEME = "postgresql+psycopg"

# Schemes we accept and rewrite. Anything else is rejected explicitly.
_ACCEPTED = {"postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2"}


def normalize_database_url(url: str) -> str:
    """Return ``url`` with its scheme rewritten to ``postgresql+psycopg``.

    Hosted providers (Supabase, Railway, Heroku-style) often hand out
    ``postgres://`` or ``postgresql://`` URLs; SQLAlchemy needs the driver named.
    """
    scheme, sep, rest = url.strip().partition("://")
    if not sep or scheme.lower() not in _ACCEPTED:
        # Never echo the URL itself: it may contain a password.
        raise ValueError("DATABASE_URL must be a PostgreSQL URL (postgres:// or postgresql://)")
    return f"{DRIVER_SCHEME}://{rest}"
