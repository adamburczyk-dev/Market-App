from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from trading_common.schemas import Interval, OHLCVBar

from src.api.deps import get_service
from src.core.engine import BacktestParams
from src.core.service import BacktestService
from src.events.publisher import NullPublisher


def make_bars(closes: list[float], symbol: str = "AAPL") -> list[OHLCVBar]:
    """Build a chronological OHLCVBar series from a list of closes."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = []
    for i, c in enumerate(closes):
        bars.append(
            OHLCVBar(
                symbol=symbol,
                timestamp=start + timedelta(days=i),
                interval=Interval.D1,
                open=c,
                high=c * 1.01,
                low=c * 0.99,
                close=c,
                volume=1_000.0,
            )
        )
    return bars


def trending_closes(
    n: int = 320, drift: float = 0.0015, vol: float = 0.005, seed: int = 7
) -> list[float]:
    """A strongly upward-drifting price path.

    Drift dominates vol (daily info-ratio ~0.3) so momentum-long is reliably
    profitable across the full sample *and* the trailing OOS window — needed for
    deterministic test assertions on Sharpe sign.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n)
    return [float(x) for x in 100.0 * np.cumprod(1.0 + rets)]


class FakeMarketDataClient:
    def __init__(self, bars: list[OHLCVBar]) -> None:
        self.bars = bars
        self.calls: list[tuple[str, Interval, int]] = []

    async def get_ohlcv(self, symbol: str, interval: Interval, limit: int = 500) -> list[OHLCVBar]:
        self.calls.append((symbol, interval, limit))
        return self.bars[-limit:]

    async def health_ok(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


def build_service(bars=None, publisher=None, **kwargs):  # type: ignore[no-untyped-def]
    return BacktestService(
        FakeMarketDataClient(bars if bars is not None else make_bars(trending_closes())),
        publisher or NullPublisher(),
        default_params=kwargs.pop("default_params", BacktestParams()),
        **kwargs,
    )


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, BacktestService]]:
    from src.main import app

    service = build_service(publisher=NullPublisher())
    app.dependency_overrides[get_service] = lambda: service
    app.state.service = service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, service
    finally:
        app.dependency_overrides.clear()
        if hasattr(app.state, "service"):
            delattr(app.state, "service")
