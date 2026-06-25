"""Pytest fixtures dla feature-engine service."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from trading_common.schemas import Interval, OHLCVBar

from src.api.deps import get_service
from src.core.service import FeatureEngineService
from src.core.store import FeatureStore
from src.events.publisher import NullPublisher
from src.main import app


def make_bars(
    n: int = 30, symbol: str = "AAPL", interval: Interval = Interval.D1
) -> list[OHLCVBar]:
    """Syntetyczne bary: lekki trend wzrostowy z oscylacją (cechy są niezdegenerowane)."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    bars: list[OHLCVBar] = []
    for i in range(n):
        close = round(100 + i * 0.5 + (0.6 if i % 2 else -0.6), 2)
        bars.append(
            OHLCVBar(
                symbol=symbol,
                timestamp=base + timedelta(days=i),
                interval=interval,
                open=close - 0.5,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1_000_000.0 + i * 1000,
                source="test",
            )
        )
    return bars


class FakeMarketDataClient:
    def __init__(self, n: int = 30) -> None:
        self.n = n
        self.calls: list[tuple] = []

    async def get_ohlcv(self, symbol, interval, limit=250):  # type: ignore[no-untyped-def]
        self.calls.append((symbol, interval, limit))
        return make_bars(self.n, symbol=symbol, interval=interval)


class FailingMarketDataClient:
    async def get_ohlcv(self, symbol, interval, limit=250):  # type: ignore[no-untyped-def]
        raise httpx.ConnectError("market-data unreachable")


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """Plain client bez podpiętego serwisu (testy health)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _build_service(market_client) -> FeatureEngineService:  # type: ignore[no-untyped-def]
    return FeatureEngineService(market_client, FeatureStore(), NullPublisher(), min_bars=20)


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, FeatureEngineService]]:
    service = _build_service(FakeMarketDataClient(n=30))
    app.dependency_overrides[get_service] = lambda: service
    app.state.service = service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, service
    finally:
        app.dependency_overrides.clear()
        if hasattr(app.state, "service"):
            delattr(app.state, "service")


@pytest.fixture
async def wired_failing() -> AsyncIterator[AsyncClient]:
    service = _build_service(FailingMarketDataClient())
    app.dependency_overrides[get_service] = lambda: service
    app.state.service = service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()
        if hasattr(app.state, "service"):
            delattr(app.state, "service")
