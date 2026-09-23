from collections.abc import Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.health import get_db_check
from app.core.config import Settings
from app.main import create_app


def make_settings(**overrides: object) -> Settings:
    """Settings isolated from the developer's real environment and .env file."""
    base: dict[str, object] = {
        "app_env": "test",
        "database_url": None,
        "cors_origins": ["http://localhost:3000"],
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


@pytest.fixture
def app() -> FastAPI:
    return create_app(make_settings())


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def override_db_check(app: FastAPI) -> Callable[[Callable[[], None]], None]:
    def _set(check: Callable[[], None]) -> None:
        app.dependency_overrides[get_db_check] = lambda: check

    return _set
