"""Testy orkiestracji RiskMgmtService."""

import pytest
from trading_common.events import CircuitBreakerLevel, EventType, SignalGeneratedEvent

from src.core.portfolio import PortfolioState

from .conftest import build_service


def signal(
    side: str = "BUY", price: float = 100.0, stop_loss: float | None = 95.0
) -> SignalGeneratedEvent:
    return SignalGeneratedEvent(
        symbol="AAPL",
        strategy_name="momentum_rank",
        signal=side,
        confidence=0.9,
        price=price,
        stop_loss=stop_loss,
        take_profit=110.0,
    )


@pytest.mark.asyncio
async def test_buy_signal_becomes_sized_order():
    service = build_service()
    order = await service.process_signal(signal())
    assert order is not None
    assert order.event_type == EventType.ORDER_REQUESTED
    assert order.side == "BUY"
    assert order.quantity == 50.0  # sized down to the 5% position cap
    assert order.stop_loss == 95.0


@pytest.mark.asyncio
async def test_hold_signal_no_order():
    service = build_service()
    assert await service.process_signal(signal(side="HOLD")) is None


@pytest.mark.asyncio
async def test_signal_without_stop_loss_blocked():
    service = build_service()
    assert await service.process_signal(signal(stop_loss=None)) is None


@pytest.mark.asyncio
async def test_blocked_when_breaker_tripped():
    service = build_service()
    await service.update_portfolio(daily_loss_pct=0.06)  # RED → tripped
    assert service.breaker.is_tripped is True
    assert await service.process_signal(signal()) is None


@pytest.mark.asyncio
async def test_blocked_by_regime_exposure_cap():
    service = build_service(portfolio=PortfolioState(exposure_pct=0.95, regime="expansion"))
    assert await service.process_signal(signal()) is None


@pytest.mark.asyncio
async def test_update_portfolio_trips_and_publishes_breaker():
    from src.events.publisher import NullPublisher

    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    event = await service.update_portfolio(daily_loss_pct=0.06)
    assert event is not None
    assert event.event_type == EventType.CIRCUIT_BREAKER_TRIGGERED
    assert event.level == CircuitBreakerLevel.RED
    assert event in publisher.published


def aggregated(
    side: str = "BUY",
    price: float | None = 100.0,
    stop_loss: float | None = 95.0,
    strategy_name: str | None = "momentum_rank",
):
    from trading_common.events import SignalAggregatedEvent

    return SignalAggregatedEvent(
        symbol="AAPL",
        final_signal=side,
        confidence=0.8,
        components_count=2,
        price=price,
        stop_loss=stop_loss,
        take_profit=110.0,
        strategy_name=strategy_name,
    )


@pytest.mark.asyncio
async def test_handle_aggregated_event_publishes_order():
    from src.events.publisher import NullPublisher

    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.handle_aggregated_event(aggregated().model_dump_json().encode())
    assert len(publisher.published) == 1
    order = publisher.published[0]
    assert order.event_type == EventType.ORDER_REQUESTED
    assert order.stop_loss == 95.0
    assert order.strategy_name == "momentum_rank"


@pytest.mark.asyncio
async def test_aggregated_hold_produces_no_order():
    service = build_service()
    assert await service.process_aggregated(aggregated(side="HOLD")) is None


@pytest.mark.asyncio
async def test_aggregated_without_levels_blocked():
    # defense-in-depth: an actionable aggregate missing price/SL cannot become an order
    service = build_service()
    assert await service.process_aggregated(aggregated(price=None)) is None
    assert await service.process_aggregated(aggregated(stop_loss=None)) is None


@pytest.mark.asyncio
async def test_aggregated_without_strategy_name_defaults():
    from src.events.publisher import NullPublisher

    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    order = await service.process_aggregated(aggregated(strategy_name=None))
    assert order is not None
    assert order.strategy_name == "aggregated"


def regime_event(old: str = "expansion", new: str = "crisis"):  # type: ignore[no-untyped-def]
    from trading_common.events import RegimeChangedEvent

    return RegimeChangedEvent(old_regime=old, new_regime=new)


@pytest.mark.asyncio
async def test_handle_regime_changed_updates_portfolio():
    service = build_service()
    await service.handle_regime_changed_event(regime_event().model_dump_json().encode())
    assert service.portfolio.regime == "crisis"


@pytest.mark.asyncio
async def test_regime_change_alone_does_not_trip_breaker():
    from src.events.publisher import NullPublisher

    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.handle_regime_changed_event(regime_event().model_dump_json().encode())
    # regime doesn't affect drawdown/daily-loss → breaker stays untripped, no event
    assert service.breaker.is_tripped is False
    assert not any(e.event_type == EventType.CIRCUIT_BREAKER_TRIGGERED for e in publisher.published)


@pytest.mark.asyncio
async def test_regime_change_persists_state():
    from .test_repository import FakeRepository

    repo = FakeRepository()
    service = build_service(repository=repo)
    payload = regime_event(new="contraction").model_dump_json().encode()
    await service.handle_regime_changed_event(payload)
    assert repo.saved[-1]["regime"] == "contraction"


@pytest.mark.asyncio
async def test_crisis_regime_tightens_exposure_cap():
    # After a crisis regime change, the regime allocator caps equity at 15% →
    # a fresh BUY that would exceed it is blocked by sizing.
    service = build_service(portfolio=PortfolioState(exposure_pct=0.30, regime="expansion"))
    # expansion allows 90% → order goes through
    assert await service.process_signal(signal()) is not None
    await service.handle_regime_changed_event(regime_event(new="crisis").model_dump_json().encode())
    # crisis caps at 15%, current exposure 30% already exceeds → blocked
    assert await service.process_signal(signal()) is None
