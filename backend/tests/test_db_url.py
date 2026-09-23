import pytest

from app.db.url import normalize_database_url

TARGET = "postgresql+psycopg://u:p@host:5432/db"


@pytest.mark.parametrize(
    "url",
    [
        "postgres://u:p@host:5432/db",
        "postgresql://u:p@host:5432/db",
        "postgresql+psycopg2://u:p@host:5432/db",
        "postgresql+psycopg://u:p@host:5432/db",
        "POSTGRES://u:p@host:5432/db",
        "  postgresql://u:p@host:5432/db  ",
    ],
)
def test_normalizes_postgres_schemes(url):
    assert normalize_database_url(url) == TARGET


def test_keeps_query_string():
    url = "postgresql://u:p@host:6543/db?sslmode=require"
    assert normalize_database_url(url) == "postgresql+psycopg://u:p@host:6543/db?sslmode=require"


@pytest.mark.parametrize(
    "url",
    [
        "mysql://u:secretpw@host/db",
        "sqlite:///x.db",
        "not-a-url-secretpw",
        "postgresql+asyncpg://u:secretpw@h/db",
    ],
)
def test_rejects_non_postgres_without_echoing_url(url):
    with pytest.raises(ValueError) as exc:
        normalize_database_url(url)
    assert "secretpw" not in str(exc.value)
