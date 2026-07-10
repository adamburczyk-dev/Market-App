"""Tests for the scheduled weekly revalidation wiring pattern.

The scheduler itself (PeriodicTask) is tested in trading-common; here we prove
the job body — a revalidation closure over BacktestService — fires end-to-end
and publishes StrategyRevalidatedEvent, exactly as main.py wires it.
"""

import asyncio

import pytest
from trading_common.events import EventType
from trading_common.scheduler import PeriodicTask
from trading_common.schemas import Interval

from src.core.service import BacktestService
from src.events.publisher import NullPublisher

from .conftest import FakeMarketDataClient, make_bars, trending_closes


@pytest.mark.asyncio
async def test_scheduled_revalidation_publishes_event():
    publisher = NullPublisher()
    service = BacktestService(FakeMarketDataClient(make_bars(trending_closes())), publisher)

    async def weekly_revalidation() -> None:
        await service.revalidate("momentum_rank", "AAPL", 1.0, Interval.D1)

    task = PeriodicTask(
        "weekly-revalidation", interval_s=60.0, job=weekly_revalidation, initial_delay_s=0.01
    )
    task.start()
    await asyncio.sleep(0.3)
    await task.stop()

    revalidated = [e for e in publisher.published if e.event_type == EventType.STRATEGY_REVALIDATED]
    assert len(revalidated) == 1  # fired once, next run a minute away
    assert revalidated[0].strategy_name == "momentum_rank"
    assert revalidated[0].recommended_status in {"active", "probation", "deactivate"}


@pytest.mark.asyncio
async def test_failing_market_data_does_not_kill_the_schedule():
    class DownMarket:
        async def get_ohlcv(self, symbol, interval, limit=500):  # type: ignore[no-untyped-def]
            raise ConnectionError("market-data down")

        async def health_ok(self) -> bool:
            return False

    service = BacktestService(DownMarket(), NullPublisher())

    async def weekly_revalidation() -> None:
        await service.revalidate("momentum_rank", "AAPL", 1.0, Interval.D1)

    task = PeriodicTask(
        "weekly-revalidation", interval_s=0.02, job=weekly_revalidation, initial_delay_s=0.0
    )
    task.start()
    await asyncio.sleep(0.08)
    assert task.running  # failed runs are isolated; the schedule survives
    await task.stop()
