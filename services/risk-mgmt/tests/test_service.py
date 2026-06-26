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


@pytest.mark.asyncio
async def test_handle_signal_event_publishes_order():
    from src.events.publisher import NullPublisher

    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.handle_signal_event(signal().model_dump_json().encode())
    assert len(publisher.published) == 1
    assert publisher.published[0].event_type == EventType.ORDER_REQUESTED
