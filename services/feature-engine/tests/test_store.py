"""Testy InMemoryFeatureStore (async interfejs)."""

from datetime import UTC, datetime

import pytest
from trading_common.schemas import FeatureVector, Interval

from src.core.store import InMemoryFeatureStore


def _fv(symbol: str, interval: Interval = Interval.D1) -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        interval=interval,
        features={"x": 1.0},
        tier=1,
    )


@pytest.mark.asyncio
async def test_put_get_roundtrip():
    store = InMemoryFeatureStore()
    fv = _fv("AAPL")
    await store.put(fv)
    assert await store.get("AAPL", Interval.D1) is fv
    assert await store.get("MSFT", Interval.D1) is None


@pytest.mark.asyncio
async def test_all_for_interval_filters():
    store = InMemoryFeatureStore()
    await store.put(_fv("AAPL", Interval.D1))
    await store.put(_fv("MSFT", Interval.H1))
    d1 = await store.all_for_interval(Interval.D1)
    assert [v.symbol for v in d1] == ["AAPL"]


@pytest.mark.asyncio
async def test_symbols_dedups_across_intervals():
    store = InMemoryFeatureStore()
    await store.put(_fv("AAPL", Interval.D1))
    await store.put(_fv("AAPL", Interval.H1))
    await store.put(_fv("MSFT", Interval.D1))
    assert await store.symbols() == ["AAPL", "MSFT"]
