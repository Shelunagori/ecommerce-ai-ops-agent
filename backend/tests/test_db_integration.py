"""Optional: runs only when TEST_DATABASE_URL points at a disposable PostgreSQL."""

import os

import pytest
from sqlalchemy import create_engine, text

from app.db.url import normalize_database_url

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set"),
]


def test_select_one_and_vector_extension_available():
    engine = create_engine(normalize_database_url(TEST_DATABASE_URL or ""))
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar_one() == 1
        available = conn.execute(
            text("SELECT count(*) FROM pg_available_extensions WHERE name = 'vector'")
        ).scalar_one()
        assert available == 1
    engine.dispose()
