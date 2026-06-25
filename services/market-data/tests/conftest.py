"""Pytest fixtures dla market-data service."""

# ruff: noqa: I001, E402
# Env vars muszą być ustawione PRZED importem src.config,
# bo Settings() jest instancjonowany na poziomie modułu.
import os

os.environ.setdefault("DB_PASSWORD", "test_password")
os.environ.setdefault("REDIS_PASSWORD", "test_redis")

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from trading_common.schemas import Interval, OHLCVBar

from src.api.deps import get_service
from src.core.cache import InMemoryCache
from src.core.fetchers.base import Fetcher
from src.core.service import MarketDataService
from src.core.storage import OHLCVRepository
from src.events.publisher import NullPublisher
from src.main import app
from src.models.db import Base


def make_bar(close: float = 100.0, day: int = 1, symbol: str = "AAPL") -> OHLCVBar:
    return OHLCVBar(
        symbol=symbol,
        timestamp=datetime(2024, 1, day, tzinfo=UTC),
        interval=Interval.D1,
        open=close - 1,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=1_000_000.0,
        source="test",
    )


class FakeFetcher(Fetcher):
    """Returns a fixed list of bars; records calls."""

    name = "fake"

    def __init__(self, bars: list[OHLCVBar]) -> None:
        self.bars = bars
        self.calls: list[tuple] = []

    async def fetch(self, symbol, interval, start=None, end=None):  # type: ignore[no-untyped-def]
        self.calls.append((symbol, interval, start, end))
        return [b.model_copy(update={"symbol": symbol}) for b in self.bars]


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """AsyncClient bez podpiętego serwisu (testy health / fallback)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def sessionmaker() -> AsyncIterator[async_sessionmaker]:
    """In-memory SQLite sessionmaker z utworzonymi tabelami."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
async def repository(sessionmaker: async_sessionmaker) -> OHLCVRepository:
    return OHLCVRepository(sessionmaker)


@pytest.fixture
async def wired(
    repository: OHLCVRepository,
) -> AsyncIterator[tuple[AsyncClient, MarketDataService]]:
    """Client z w pełni podpiętym serwisem (FakeFetcher + SQLite + in-memory cache)."""
    bars = [make_bar(close=c, day=d) for d, c in enumerate([101, 102, 103], start=1)]
    service = MarketDataService(FakeFetcher(bars), repository, InMemoryCache(), NullPublisher())

    app.dependency_overrides[get_service] = lambda: service
    app.state.service = service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, service
    finally:
        app.dependency_overrides.clear()
        if hasattr(app.state, "service"):
            delattr(app.state, "service")
