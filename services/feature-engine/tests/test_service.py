"""Testy orkiestracji FeatureEngineService."""

import pytest
from trading_common.events import EventType, MarketDataUpdatedEvent
from trading_common.schemas import Interval

from src.core.service import FeatureEngineService
from src.core.store import InMemoryFeatureStore
from src.events.publisher import NullPublisher

from .conftest import FakeMarketDataClient


@pytest.mark.asyncio
async def test_compute_stores_and_publishes():
    publisher = NullPublisher()
    service = FeatureEngineService(
        FakeMarketDataClient(n=30), InMemoryFeatureStore(), publisher, min_bars=20
    )

    fv = await service.compute_for_symbol("AAPL", Interval.D1)
    assert fv is not None
    assert await service.get_features("AAPL", Interval.D1) is fv

    assert len(publisher.published) == 1
    event = publisher.published[0]
    assert event.event_type == EventType.FEATURES_READY
    assert event.symbol == "AAPL"  # type: ignore[attr-defined]
    assert event.features_count == len(fv.features)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_insufficient_bars_returns_none_and_no_event():
    publisher = NullPublisher()
    service = FeatureEngineService(
        FakeMarketDataClient(n=5), InMemoryFeatureStore(), publisher, min_bars=20
    )
    fv = await service.compute_for_symbol("AAPL", Interval.D1)
    assert fv is None
    assert publisher.published == []


@pytest.mark.asyncio
async def test_handle_market_data_event_triggers_compute():
    publisher = NullPublisher()
    service = FeatureEngineService(
        FakeMarketDataClient(n=30), InMemoryFeatureStore(), publisher, min_bars=20
    )

    event = MarketDataUpdatedEvent(symbol="MSFT", interval="1d", rows_count=30)
    await service.handle_market_data_event(event.model_dump_json().encode())

    assert await service.get_features("MSFT", Interval.D1) is not None
    assert await service.list_symbols() == ["MSFT"]
    assert len(publisher.published) == 1
