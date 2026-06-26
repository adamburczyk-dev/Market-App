"""Pytest fixtures dla strategy service."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from trading_common.risk_envelope import RiskEnvelope
from trading_common.schemas import FeatureVector, Interval

from src.api.deps import get_service
from src.core.cost_filter import CostAwareFilter
from src.core.health import StrategyHealthTracker
from src.core.momentum import MomentumParams
from src.core.service import PortfolioSnapshot, StrategyService
from src.events.publisher import NullPublisher
from src.main import app


def fv(symbol: str, features: dict[str, float], interval: Interval = Interval.D1) -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        interval=interval,
        features=features,
        tier=1,
    )


class FakeFeatureClient:
    """Returns configurable ranked / raw feature vectors for any symbol."""

    def __init__(self, ranked: dict[str, float], raw: dict[str, float]) -> None:
        self.ranked = ranked
        self.raw = raw

    async def get_ranked(self, symbol, interval):  # type: ignore[no-untyped-def]
        return fv(symbol, self.ranked, interval)

    async def get_features(self, symbol, interval):  # type: ignore[no-untyped-def]
        return fv(symbol, self.raw, interval)


class FailingFeatureClient:
    async def get_ranked(self, symbol, interval):  # type: ignore[no-untyped-def]
        raise httpx.ConnectError("feature-engine down")

    async def get_features(self, symbol, interval):  # type: ignore[no-untyped-def]
        raise httpx.ConnectError("feature-engine down")


def build_service(
    client,  # type: ignore[no-untyped-def]
    publisher=None,
    portfolio=None,
    expected_edge_bps: float = 200.0,
    name: str = "momentum_rank",
) -> StrategyService:
    return StrategyService(
        client,
        publisher or NullPublisher(),
        StrategyHealthTracker(name),
        RiskEnvelope(),
        CostAwareFilter(),
        MomentumParams(),
        portfolio or PortfolioSnapshot(),
        strategy_name=name,
        expected_edge_bps=expected_edge_bps,
    )


def buy_client() -> FakeFeatureClient:
    # top-decile momentum, neutral RSI -> BUY
    return FakeFeatureClient(ranked={"momentum_20": 0.9}, raw={"rsi_14": 50.0, "close": 100.0})


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, StrategyService]]:
    service = build_service(buy_client())
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
    service = build_service(FailingFeatureClient())
    app.dependency_overrides[get_service] = lambda: service
    app.state.service = service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()
        if hasattr(app.state, "service"):
            delattr(app.state, "service")
