"""N1: BLACK must close the book, not just raise an alarm.

The circuit breaker published CircuitBreakerTriggeredEvent(action="flatten_all")
and only notification consumed it. The non-negotiable rule "drawdown > 15% →
flatten all positions" was therefore an alert with no actor behind it: every
position stayed open through the worst drawdown the system can detect.
"""

import pytest
from trading_common.events import (
    CircuitBreakerLevel,
    CircuitBreakerTriggeredEvent,
    EventType,
    OrderRequestedEvent,
)

from src.core.paper_broker import PaperBroker
from src.events.publisher import NullPublisher

from .conftest import build_service


def black_event(action: str = "flatten_all") -> bytes:
    return (
        CircuitBreakerTriggeredEvent(
            level=CircuitBreakerLevel.BLACK,
            trigger_metric="drawdown",
            current_value=0.17,
            threshold_value=0.15,
            action_taken=action,
        )
        .model_dump_json()
        .encode()
    )


async def open_two_positions(service) -> None:  # type: ignore[no-untyped-def]
    for symbol, price in (("AAPL", 100.0), ("MSFT", 200.0)):
        await service.execute(
            OrderRequestedEvent(
                symbol=symbol,
                side="BUY",
                quantity=10,
                price=price,
                strategy_name="momentum",
                stop_loss=price * 0.9,
            )
        )


@pytest.mark.asyncio
async def test_black_event_closes_every_position():
    publisher = NullPublisher()
    service = build_service(publisher=publisher, broker=PaperBroker(initial_cash=100_000))
    await open_two_positions(service)
    assert len(service.broker.positions()) == 2

    await service.handle_circuit_breaker_event(black_event())

    assert service.broker.positions() == {}, "BLACK left positions open"
    exits = [
        e
        for e in publisher.published
        if e.event_type == EventType.ORDER_FILLED and e.order_id.startswith("liquidate-")
    ]
    assert {e.symbol for e in exits} == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_other_breaker_levels_do_not_liquidate():
    """RED halts NEW orders for the day; it must not close the book."""
    publisher = NullPublisher()
    service = build_service(publisher=publisher, broker=PaperBroker(initial_cash=100_000))
    await open_two_positions(service)

    await service.handle_circuit_breaker_event(black_event(action="halt_trading"))

    assert len(service.broker.positions()) == 2


@pytest.mark.asyncio
async def test_flatten_is_idempotent_and_safe_when_flat():
    publisher = NullPublisher()
    service = build_service(publisher=publisher, broker=PaperBroker(initial_cash=100_000))
    await open_two_positions(service)

    first = await service.flatten_all(reason="test")
    second = await service.flatten_all(reason="test")  # redelivery of the same event

    assert len(first) == 2
    assert second == []  # nothing left to close, and no exception
    assert service.broker.positions() == {}


@pytest.mark.asyncio
async def test_flatten_updates_cash_and_pushes_portfolio():
    class RecordingRisk:
        def __init__(self) -> None:
            self.pushes: list[dict] = []

        async def push_portfolio(self, metrics: dict) -> None:
            self.pushes.append(metrics)

        async def aclose(self) -> None:
            return None

    risk = RecordingRisk()
    service = build_service(risk_client=risk, broker=PaperBroker(initial_cash=100_000))
    await open_two_positions(service)
    cash_before = service.broker.cash

    await service.handle_circuit_breaker_event(black_event())

    assert service.broker.cash > cash_before  # proceeds returned to cash
    assert risk.pushes, "risk-mgmt was never told the book is flat"
