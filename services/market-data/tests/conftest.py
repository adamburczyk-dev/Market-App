"""Pytest fixtures dla market-data service."""

# Env vars muszą być ustawione PRZED importem src.config,
# bo Settings() jest instancjonowany na poziomie modułu.
import os
os.environ.setdefault("DB_PASSWORD", "test_password")
os.environ.setdefault("REDIS_PASSWORD", "test_redis")

import pytest
from httpx import AsyncClient, ASGITransport

from src.main import app


@pytest.fixture
async def client() -> AsyncClient:
    """AsyncClient do testowania endpointów FastAPI."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
