"""Testy orkiestracji MarketDataService."""

import pytest
from trading_common.events import EventType
from trading_common.schemas import Interval

from src.core.cache import InMemoryCache
from src.core.service import MarketDataService
from src.core.storage import OHLCVRepository
from src.events.publisher import NullPublisher

from .conftest import FakeFetcher, make_bar


@pytest.mark.asyncio
async def test_fetch_and_store_persists_and_publishes(repository: OHLCVRepository):
    fetcher = FakeFetcher([make_bar(close=c, day=d) for d, c in enumerate([10, 11], start=1)])
    publisher = NullPublisher()
    service = MarketDataService(fetcher, repository, InMemoryCache(), publisher)

    count = await service.fetch_and_store("AAPL", Interval.D1)
    assert count == 2

    # zapisane w storage
    stored = await service.get_ohlcv("AAPL", Interval.D1)
    assert len(stored) == 2

    # opublikowano zdarzenie MarketDataUpdated
    assert len(publisher.published) == 1
    event = publisher.published[0]
    assert event.event_type == EventType.MARKET_DATA_UPDATED
    assert event.rows_count == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_no_event_when_nothing_fetched(repository: OHLCVRepository):
    publisher = NullPublisher()
    service = MarketDataService(FakeFetcher([]), repository, InMemoryCache(), publisher)
    count = await service.fetch_and_store("AAPL", Interval.D1)
    assert count == 0
    assert publisher.published == []


@pytest.mark.asyncio
async def test_get_ohlcv_uses_cache(repository: OHLCVRepository):
    cache = InMemoryCache()
    service = MarketDataService(FakeFetcher([]), repository, cache, NullPublisher())
    await repository.save_bars([make_bar(close=50, day=1)])

    # pierwszy odczyt zapełnia cache
    first = await service.get_ohlcv("AAPL", Interval.D1)
    assert len(first) == 1
    cached = await cache.get_bars("AAPL", Interval.D1)
    assert cached is not None and len(cached) == 1


@pytest.mark.asyncio
async def test_fetch_and_store_invalidates_cache(repository: OHLCVRepository):
    cache = InMemoryCache()
    fetcher = FakeFetcher([make_bar(close=10, day=1)])
    service = MarketDataService(fetcher, repository, cache, NullPublisher())

    await repository.save_bars([make_bar(close=50, day=1)])
    await service.get_ohlcv("AAPL", Interval.D1)  # zapełnia cache
    assert await cache.get_bars("AAPL", Interval.D1) is not None

    await service.fetch_and_store("AAPL", Interval.D1)
    assert await cache.get_bars("AAPL", Interval.D1) is None  # cache zinwalidowany


@pytest.mark.asyncio
async def test_short_cached_window_does_not_shrink_a_larger_request(
    repository: OHLCVRepository,
):
    """A cached window may only answer requests it actually covers.

    The cache key is (symbol, interval) and carries no limit, while the cached
    list was produced by some earlier limit. Serving a bigger request from a
    shorter cached window silently returns fewer bars than asked for — that is
    how a feature-engine read (limit=250) starved a training read (limit=2000)
    down to 250 bars, and the model saw a fraction of the stored history.
    """
    bars = [make_bar(close=100 + i, day=i + 1) for i in range(28)]
    await repository.save_bars(bars)
    cache = InMemoryCache()
    service = MarketDataService(FakeFetcher([]), repository, cache, NullPublisher())

    small = await service.get_ohlcv("AAPL", Interval.D1, limit=5)
    assert len(small) == 5  # warms the cache with a SHORT window

    large = await service.get_ohlcv("AAPL", Interval.D1, limit=20)
    assert len(large) == 20, "larger request was served from the shorter cached window"

    # the longer window now backs the cache, so the small read stays correct
    again = await service.get_ohlcv("AAPL", Interval.D1, limit=5)
    assert len(again) == 5
    assert [b.close for b in again] == [b.close for b in small]
