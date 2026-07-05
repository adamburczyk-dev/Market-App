"""Tests for the live, event-driven aggregation path (NATS handlers + buffer)."""

import pytest
from trading_common.events import EventType, RegimeChangedEvent, SignalGeneratedEvent

from src.core.aggregator import regime_to_component
from src.events.publisher import NullPublisher

from .conftest import build_service


def signal_event(symbol: str = "AAPL", side: str = "BUY", confidence: float = 0.9):
    return SignalGeneratedEvent(
        symbol=symbol,
        strategy_name="momentum_rank",
        signal=side,
        confidence=confidence,
        price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
    )


def regime_event(old: str = "expansion", new: str = "crisis"):
    return RegimeChangedEvent(old_regime=old, new_regime=new)


class TestRegimeBias:
    def test_all_regimes_map_to_direction(self):
        assert regime_to_component("expansion").signal == "BUY"
        assert regime_to_component("recovery").signal == "BUY"
        assert regime_to_component("slowdown").signal == "HOLD"
        assert regime_to_component("contraction").signal == "SELL"
        assert regime_to_component("crisis").signal == "SELL"

    def test_unknown_regime_is_none(self):
        assert regime_to_component("weird") is None

    def test_crisis_bias_stronger_than_contraction(self):
        crisis = regime_to_component("crisis")
        contraction = regime_to_component("contraction")
        assert crisis.confidence > contraction.confidence


@pytest.mark.asyncio
async def test_signal_event_buffers_and_aggregates():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.handle_signal_generated(signal_event().model_dump_json().encode())
    assert len(publisher.published) == 1
    event = publisher.published[0]
    assert event.event_type == EventType.SIGNAL_AGGREGATED
    assert event.symbol == "AAPL"
    # single present source renormalizes to weight 1.0 → strategy BUY passes through
    assert event.final_signal == "BUY"
    assert event.components_count == 1


@pytest.mark.asyncio
async def test_regime_change_reaggregates_all_buffered_symbols():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.handle_signal_generated(signal_event("AAPL").model_dump_json().encode())
    await service.handle_signal_generated(signal_event("MSFT").model_dump_json().encode())
    publisher.published.clear()

    await service.handle_regime_changed(regime_event().model_dump_json().encode())
    assert len(publisher.published) == 2
    assert {e.symbol for e in publisher.published} == {"AAPL", "MSFT"}
    # each re-aggregation now includes the macro component alongside strategy
    assert all(e.components_count == 2 for e in publisher.published)


@pytest.mark.asyncio
async def test_crisis_bias_dampens_strategy_buy_to_hold():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.handle_regime_changed(regime_event(new="crisis").model_dump_json().encode())
    await service.handle_signal_generated(signal_event(confidence=0.9).model_dump_json().encode())
    event = publisher.published[-1]
    # 0.5*0.9 (BUY) − 0.5*0.6 (crisis SELL bias) = 0.15 < 0.2 threshold → HOLD
    assert event.final_signal == "HOLD"
    assert event.components_count == 2


@pytest.mark.asyncio
async def test_expansion_bias_supports_buy():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    payload = regime_event(old="crisis", new="expansion").model_dump_json().encode()
    await service.handle_regime_changed(payload)
    await service.handle_signal_generated(signal_event(confidence=0.9).model_dump_json().encode())
    event = publisher.published[-1]
    # 0.5*0.9 + 0.5*0.5 = 0.7 → BUY
    assert event.final_signal == "BUY"


@pytest.mark.asyncio
async def test_regime_change_with_empty_buffer_publishes_nothing():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.handle_regime_changed(regime_event().model_dump_json().encode())
    assert publisher.published == []


@pytest.mark.asyncio
async def test_aggregate_symbol_none_when_not_buffered():
    service = build_service()
    assert await service.aggregate_symbol("GHOST") is None


@pytest.mark.asyncio
async def test_new_signal_replaces_previous_component():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.handle_signal_generated(signal_event(side="BUY").model_dump_json().encode())
    await service.handle_signal_generated(
        signal_event(side="SELL", confidence=0.8).model_dump_json().encode()
    )
    # latest-per-source wins: second aggregation reflects the SELL, not a mix
    assert publisher.published[-1].final_signal == "SELL"
    assert publisher.published[-1].components_count == 1
