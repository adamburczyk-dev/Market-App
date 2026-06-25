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
