"""Tests for the live, event-driven aggregation path (NATS handlers + buffer)."""

from datetime import datetime

import pytest
from trading_common.events import EventType, RegimeChangedEvent, SignalGeneratedEvent

from src.core.aggregator import regime_to_component
from src.events.publisher import NullPublisher

from .conftest import build_service


def signal_event(
    symbol: str = "AAPL",
    side: str = "BUY",
    confidence: float = 0.9,
    ts: datetime | None = None,
):
    kwargs = {"timestamp": ts} if ts is not None else {}
    return SignalGeneratedEvent(
        symbol=symbol,
        strategy_name="momentum_rank",
        signal=side,
        confidence=confidence,
        price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        **kwargs,
    )


def regime_event(old: str = "expansion", new: str = "crisis"):
    return RegimeChangedEvent(old_regime=old, new_regime=new)


class TestRegimeBias:
    def test_directional_regimes_map_to_direction(self):
        assert regime_to_component("expansion").signal == "BUY"
        assert regime_to_component("recovery").signal == "BUY"
        assert regime_to_component("contraction").signal == "SELL"
        assert regime_to_component("crisis").signal == "SELL"

    def test_slowdown_is_neutral_and_contributes_nothing(self):
        # R10: a known-neutral regime must not claim weight (a HOLD component
        # would dilute the strategy signal more than an unknown regime does)
        assert regime_to_component("slowdown") is None

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


@pytest.mark.asyncio
async def test_actionable_aggregate_carries_order_context():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.handle_signal_generated(signal_event().model_dump_json().encode())
    event = publisher.published[-1]
    assert event.final_signal == "BUY"
    assert event.price == 100.0
    assert event.stop_loss == 95.0
    assert event.take_profit == 110.0
    assert event.strategy_name == "momentum_rank"


@pytest.mark.asyncio
async def test_hold_aggregate_carries_no_levels():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.handle_regime_changed(regime_event(new="crisis").model_dump_json().encode())
    await service.handle_signal_generated(signal_event(confidence=0.9).model_dump_json().encode())
    event = publisher.published[-1]
    assert event.final_signal == "HOLD"  # crisis bias dampens the BUY
    assert event.price is None
    assert event.stop_loss is None
    assert event.take_profit is None


class FakeClock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now


@pytest.mark.asyncio
async def test_expired_signal_is_pruned_and_yields_none():
    from datetime import UTC, datetime, timedelta

    start = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
    clock = FakeClock(start)
    publisher = NullPublisher()
    service = build_service(publisher=publisher, signal_ttl_s=3600.0, clock=clock)
    await service.handle_signal_generated(signal_event(ts=start).model_dump_json().encode())
    assert len(publisher.published) == 1

    clock.now += timedelta(hours=2)  # beyond the 1h TTL
    assert await service.aggregate_symbol("AAPL") is None
    # pruned: a regime change no longer resurfaces the stale signal
    await service.handle_regime_changed(regime_event().model_dump_json().encode())
    assert len(publisher.published) == 1  # nothing new published


@pytest.mark.asyncio
async def test_fresh_signal_survives_ttl_window():
    from datetime import UTC, datetime, timedelta

    start = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
    clock = FakeClock(start)
    publisher = NullPublisher()
    service = build_service(publisher=publisher, signal_ttl_s=3600.0, clock=clock)
    await service.handle_signal_generated(signal_event(ts=start).model_dump_json().encode())
    clock.now += timedelta(minutes=30)  # within TTL
    result = await service.aggregate_symbol("AAPL")
    assert result is not None
    assert result.final_signal == "BUY"


@pytest.mark.asyncio
async def test_replayed_old_event_expires_by_emit_timestamp():
    """A durable replaying stream history must not resurrect a stale signal.

    The buffer ages entries from the event's emit timestamp, not receive time —
    a week-old signal.generated delivered *now* is already past the TTL.
    """
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 7, 7, 12, 0, tzinfo=UTC)
    clock = FakeClock(now)
    publisher = NullPublisher()
    service = build_service(publisher=publisher, signal_ttl_s=3600.0, clock=clock)
    stale = signal_event(ts=now - timedelta(days=7))  # replayed history
    await service.handle_signal_generated(stale.model_dump_json().encode())
    assert publisher.published == []  # pruned on arrival, never aggregated
    assert await service.aggregate_symbol("AAPL") is None
