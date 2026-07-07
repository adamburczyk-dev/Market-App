"""Tests for the 2026-07-05 review fixes: R2 daily rollover, R3 fill idempotency,
R4 long-only, R5 protective SL/TP exits."""

from datetime import date

import pytest
from trading_common.events import EventType, MarketDataUpdatedEvent

from src.core.paper_broker import PaperBroker
from src.core.risk_client import NullRiskClient
from src.core.service import ExecutionService
from src.events.publisher import NullPublisher

from .conftest import build_service
from .test_service import FakeMarketDataClient, order


class FakeClock:
    def __init__(self, start: date) -> None:
        self.today = start

    def __call__(self) -> date:
        return self.today


# --- R2: daily-loss rollover ---


def test_daily_loss_resets_on_new_day():
    clock = FakeClock(date(2026, 7, 5))
    broker = PaperBroker(initial_cash=100_000.0, clock=clock)
    broker.fill("o1", "AAPL", "BUY", 50, 100.0)
    broker.mark("AAPL", 90.0)  # equity 99_500 → daily loss 0.5%
    assert broker.metrics()["daily_loss_pct"] == pytest.approx(0.005)

    clock.today = date(2026, 7, 6)
    broker.mark("AAPL", 90.0)  # first event of the new day → baseline rolls to 99_500
    assert broker.metrics()["daily_loss_pct"] == 0.0
    # a further drop counts against the NEW baseline
    broker.mark("AAPL", 80.0)  # equity 99_000 vs 99_500
    assert broker.metrics()["daily_loss_pct"] == pytest.approx(500 / 99_500)


def test_day_start_date_survives_snapshot_round_trip():
    clock = FakeClock(date(2026, 7, 5))
    broker = PaperBroker(initial_cash=100_000.0, clock=clock)
    broker.fill("o1", "AAPL", "BUY", 50, 100.0)
    snap = broker.snapshot()
    assert snap["day_start_date"] == "2026-07-05"

    restored = PaperBroker(initial_cash=1.0, clock=clock)
    restored.restore(snap)
    assert restored.metrics()["daily_loss_pct"] == broker.metrics()["daily_loss_pct"]


def test_restore_tolerates_old_snapshot_without_new_keys():
    broker = PaperBroker(initial_cash=100_000.0)
    broker.restore(
        {
            "cash": 95_000.0,
            "peak_equity": 100_000.0,
            "day_start_equity": 100_000.0,
            "positions": {"AAPL": {"quantity": 50.0, "last_price": 100.0}},
        }
    )
    assert broker.cash == 95_000.0
    assert broker.position_qty("AAPL") == 50.0


# --- R3: fill idempotency ---


@pytest.mark.asyncio
async def test_duplicate_order_delivery_fills_once():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    o = order()  # one event_id
    payload = o.model_dump_json().encode()
    await service.handle_order_event(payload)
    await service.handle_order_event(payload)  # redelivery
    assert service.broker.cash == 95_000.0  # single fill
    assert service.broker.position_qty("AAPL") == 50.0
    assert len(publisher.published) == 1


def test_broker_fill_duplicate_returns_none():
    broker = PaperBroker(initial_cash=100_000.0)
    assert broker.fill("o1", "AAPL", "BUY", 50, 100.0) is not None
    assert broker.fill("o1", "AAPL", "BUY", 50, 100.0) is None
    assert broker.cash == 95_000.0


def test_processed_orders_survive_snapshot_round_trip():
    broker = PaperBroker(initial_cash=100_000.0)
    broker.fill("o1", "AAPL", "BUY", 50, 100.0)
    restored = PaperBroker(initial_cash=100_000.0)
    restored.restore(broker.snapshot())
    assert restored.is_processed("o1")
    assert restored.fill("o1", "AAPL", "BUY", 50, 100.0) is None  # still deduped


# --- R4: long-only ---


@pytest.mark.asyncio
async def test_sell_without_position_is_skipped():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    result = await service.execute(order(side="SELL"))
    assert result is None
    assert publisher.published == []
    assert service.broker.cash == 100_000.0  # no naked short, no cash credit


@pytest.mark.asyncio
async def test_sell_capped_at_held_quantity():
    service = build_service()
    await service.execute(order(side="BUY", qty=50.0))
    result = await service.execute(order(side="SELL", qty=200.0, price=110.0))
    assert result is not None
    assert result.filled_quantity == 50.0  # capped — no short opened
    assert service.broker.position_qty("AAPL") == 0.0


# --- R5: protective SL/TP exits on re-mark ---


def build_marking_service(close: float):
    publisher = NullPublisher()
    risk = NullRiskClient()
    broker = PaperBroker(initial_cash=100_000.0)
    service = ExecutionService(broker, publisher, risk, FakeMarketDataClient(close=close))
    return service, broker, publisher


async def mark(service: ExecutionService) -> None:
    event = MarketDataUpdatedEvent(symbol="AAPL", interval="1d", rows_count=1)
    await service.handle_market_data_event(event.model_dump_json().encode())


@pytest.mark.asyncio
async def test_stop_loss_breach_exits_position():
    service, broker, publisher = build_marking_service(close=94.0)  # below SL 95
    await service.execute(order())  # BUY 50 @100, SL 95 / TP 110
    await mark(service)
    assert broker.position_qty("AAPL") == 0.0
    assert broker.cash == pytest.approx(95_000.0 + 50 * 94.0)
    fills = [e for e in publisher.published if e.event_type == EventType.ORDER_FILLED]
    assert len(fills) == 2  # entry + protective exit
    assert fills[-1].filled_price == 94.0


@pytest.mark.asyncio
async def test_take_profit_breach_exits_position():
    service, broker, publisher = build_marking_service(close=111.0)  # above TP 110
    await service.execute(order())
    await mark(service)
    assert broker.position_qty("AAPL") == 0.0
    assert broker.cash == pytest.approx(95_000.0 + 50 * 111.0)


@pytest.mark.asyncio
async def test_mark_between_levels_keeps_position():
    service, broker, publisher = build_marking_service(close=100.5)
    await service.execute(order())
    await mark(service)
    assert broker.position_qty("AAPL") == 50.0
    fills = [e for e in publisher.published if e.event_type == EventType.ORDER_FILLED]
    assert len(fills) == 1  # entry only


@pytest.mark.asyncio
async def test_position_without_levels_never_auto_exits():
    service, broker, _ = build_marking_service(close=10.0)  # brutal drop
    await service.execute(order(side="BUY"))
    # wipe levels to simulate a legacy position
    broker._positions["AAPL"].stop_loss = None
    broker._positions["AAPL"].take_profit = None
    await mark(service)
    assert broker.position_qty("AAPL") == 50.0  # held — no levels, no exit


def test_levels_survive_snapshot_round_trip():
    broker = PaperBroker(initial_cash=100_000.0)
    broker.fill("o1", "AAPL", "BUY", 50, 100.0, stop_loss=95.0, take_profit=110.0)
    restored = PaperBroker(initial_cash=100_000.0)
    restored.restore(broker.snapshot())
    assert restored.positions()["AAPL"]["stop_loss"] == 95.0
    assert restored.positions()["AAPL"]["take_profit"] == 110.0
    restored.mark("AAPL", 94.0)
    assert restored.protective_trigger("AAPL") == "stop_loss"


def test_exit_clears_levels():
    broker = PaperBroker(initial_cash=100_000.0)
    broker.fill("o1", "AAPL", "BUY", 50, 100.0, stop_loss=95.0, take_profit=110.0)
    broker.fill("o2", "AAPL", "SELL", 50, 96.0)
    assert broker.positions() == {}  # flat, hidden from the view
    broker.fill("o3", "AAPL", "BUY", 10, 96.0)  # fresh position, no stale levels
    assert broker.positions()["AAPL"]["stop_loss"] is None
