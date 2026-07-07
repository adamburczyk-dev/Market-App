"""Testy orkiestracji ExecutionService."""

import pytest
from trading_common.events import EventType, MarketDataUpdatedEvent, OrderRequestedEvent

from src.core.paper_broker import PaperBroker
from src.core.risk_client import NullRiskClient
from src.core.service import ExecutionService
from src.events.publisher import NullPublisher

from .conftest import build_service


class FakeMarketDataClient:
    def __init__(self, close: float) -> None:
        self.close = close

    async def latest_close(self, symbol, interval):  # type: ignore[no-untyped-def]
        return self.close


def order(side: str = "BUY", qty: float = 50.0, price: float = 100.0) -> OrderRequestedEvent:
    return OrderRequestedEvent(
        symbol="AAPL",
        side=side,
        quantity=qty,
        price=price,
        strategy_name="momentum_rank",
        stop_loss=95.0,
        take_profit=110.0,
    )


@pytest.mark.asyncio
async def test_execute_fills_and_publishes():
    publisher = NullPublisher()
    risk = NullRiskClient()
    service = build_service(publisher=publisher, risk_client=risk)

    fill = await service.execute(order())
    assert fill.event_type == EventType.ORDER_FILLED
    assert fill.symbol == "AAPL"
    assert fill.filled_quantity == 50.0
    assert fill.filled_price == 100.0
    assert len(publisher.published) == 1


@pytest.mark.asyncio
async def test_execute_pushes_portfolio_to_risk():
    risk = NullRiskClient()
    service = build_service(risk_client=risk)
    await service.execute(order())
    assert len(risk.pushed) == 1
    assert "exposure_pct" in risk.pushed[0]
    assert risk.pushed[0]["value"] == 100_000.0


@pytest.mark.asyncio
async def test_handle_order_event_fills():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.handle_order_event(order().model_dump_json().encode())
    assert len(publisher.published) == 1
    assert publisher.published[0].event_type == EventType.ORDER_FILLED
    assert service.broker.positions()["AAPL"]["quantity"] == 50.0


@pytest.mark.asyncio
async def test_market_data_event_remarks_held_position():
    risk = NullRiskClient()
    broker = PaperBroker(initial_cash=100_000.0)
    # 96 is above the order's stop_loss (95) → a pure re-mark, no protective exit
    service = ExecutionService(broker, NullPublisher(), risk, FakeMarketDataClient(close=96.0))
    await service.execute(order())  # BUY 50 AAPL @100, SL 95 / TP 110
    risk.pushed.clear()

    event = MarketDataUpdatedEvent(symbol="AAPL", interval="1d", rows_count=1)
    await service.handle_market_data_event(event.model_dump_json().encode())

    assert broker.positions()["AAPL"]["last_price"] == 96.0
    assert broker.equity == 99_800.0  # unrealized loss
    assert len(risk.pushed) == 1
    assert risk.pushed[0]["drawdown_pct"] > 0


@pytest.mark.asyncio
async def test_market_data_event_ignored_when_not_held():
    risk = NullRiskClient()
    service = ExecutionService(PaperBroker(), NullPublisher(), risk, FakeMarketDataClient(90.0))
    event = MarketDataUpdatedEvent(symbol="TSLA", interval="1d", rows_count=1)
    await service.handle_market_data_event(event.model_dump_json().encode())
    assert risk.pushed == []  # not held → no mark, no push
