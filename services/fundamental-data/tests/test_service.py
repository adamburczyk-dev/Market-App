"""Tests for FundamentalDataService — score, store, publish."""

import pytest
from trading_common.events import EventType

from src.events.publisher import NullPublisher

from .conftest import FakeFetcher, build_service, improving_pair


@pytest.mark.asyncio
async def test_ingest_scores_and_publishes():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    current, prior = improving_pair()
    scored, breakdown = await service.ingest(current, prior)
    assert scored.piotroski_f_score == 9
    assert breakdown.score == 9
    assert len(publisher.published) == 1
    event = publisher.published[0]
    assert event.event_type == EventType.FUNDAMENTALS_UPDATED
    assert event.symbol == "AAPL"
    assert event.period_end == "2024-12-31"
    assert event.fiscal_period == "FY"
    assert event.source_service == "fundamental-data"


@pytest.mark.asyncio
async def test_ingest_stores_latest_by_symbol():
    service = build_service()
    current, prior = improving_pair()
    await service.ingest(current, prior)
    record = service.get("AAPL")
    assert record is not None
    assert record[0].piotroski_f_score == 9
    assert service.symbols() == ["AAPL"]


@pytest.mark.asyncio
async def test_get_is_case_insensitive():
    service = build_service()
    current, prior = improving_pair()
    await service.ingest(current, prior)
    assert service.get("aapl") is not None


@pytest.mark.asyncio
async def test_refresh_uses_fetcher_statements():
    publisher = NullPublisher()
    current, prior = improving_pair()
    fetcher = FakeFetcher([current, prior], enabled=True)
    service = build_service(fetcher=fetcher, publisher=publisher)
    record = await service.refresh("AAPL")
    assert record is not None
    assert record[0].piotroski_f_score == 9
    assert publisher.published[0].event_type == EventType.FUNDAMENTALS_UPDATED


@pytest.mark.asyncio
async def test_refresh_returns_none_without_data():
    publisher = NullPublisher()
    service = build_service(fetcher=FakeFetcher([]), publisher=publisher)
    assert await service.refresh("AAPL") is None
    assert publisher.published == []


@pytest.mark.asyncio
async def test_refresh_single_period_scores_current_only():
    current, _ = improving_pair()
    fetcher = FakeFetcher([current], enabled=True)  # only one period available
    service = build_service(fetcher=fetcher)
    record = await service.refresh("AAPL")
    assert record is not None
    assert record[0].piotroski_f_score == 3  # trend signals can't fire


# --- scheduled universe refresh ---


@pytest.mark.asyncio
async def test_refresh_universe_counts_refreshed_symbols():
    publisher = NullPublisher()
    current, prior = improving_pair()
    fetcher = FakeFetcher([current, prior], enabled=True)
    service = build_service(fetcher=fetcher, publisher=publisher)
    refreshed = await service.refresh_universe(["AAPL", "MSFT"], pause_s=0.0)
    assert refreshed == 2  # FakeFetcher serves both symbols
    assert len(publisher.published) == 2


@pytest.mark.asyncio
async def test_refresh_universe_skips_symbols_without_data():
    service = build_service(fetcher=FakeFetcher([], enabled=True))
    assert await service.refresh_universe(["GHOST", "ZOMBIE"], pause_s=0.0) == 0


@pytest.mark.asyncio
async def test_refresh_universe_empty_list_is_noop():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    assert await service.refresh_universe([], pause_s=0.0) == 0
    assert publisher.published == []
