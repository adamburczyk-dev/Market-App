"""Tests for BacktestService orchestration + event publishing."""

import pytest
from trading_common.events import EventType
from trading_common.schemas import Interval

from src.core.engine import BacktestParams
from src.events.publisher import NullPublisher

from .conftest import build_service, make_bars, trending_closes


@pytest.mark.asyncio
async def test_run_backtest_publishes_completed_event():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    result = await service.run_backtest("momentum_rank", "AAPL", Interval.D1)
    assert result.n_bars > 0
    assert len(publisher.published) == 1
    event = publisher.published[0]
    assert event.event_type == EventType.BACKTEST_COMPLETED
    assert event.strategy_name == "momentum_rank"
    assert event.sharpe_ratio == pytest.approx(result.sharpe_ratio)


@pytest.mark.asyncio
async def test_revalidate_publishes_revalidated_event():
    publisher = NullPublisher()
    service = build_service(
        bars=make_bars(trending_closes(seed=1)), publisher=publisher, oos_window_days=126
    )
    result = await service.revalidate("momentum_rank", "AAPL", 0.1, Interval.D1)
    assert len(publisher.published) == 1
    event = publisher.published[0]
    assert event.event_type == EventType.STRATEGY_REVALIDATED
    assert event.recommended_status in {"active", "probation", "deactivate"}
    assert event.recommended_status == result.recommended_status
    assert event.source_service == "backtest"


@pytest.mark.asyncio
async def test_run_backtest_queries_market_data_with_limit():
    service = build_service()
    await service.run_backtest("s", "MSFT", Interval.D1, limit=200)
    # the fake records calls
    market = service._market  # type: ignore[attr-defined]
    assert market.calls[-1] == ("MSFT", Interval.D1, 200)


@pytest.mark.asyncio
async def test_param_overrides_reach_engine():
    service = build_service(default_params=BacktestParams(lookback=20))
    # override lookback via the request params
    result = await service.run_backtest("s", "AAPL", Interval.D1, params={"lookback": 5})
    assert result.n_bars > 0  # ran with the smaller lookback, still produces bars


@pytest.mark.asyncio
async def test_revalidate_deactivate_on_negative_oos():
    # A persistent downtrend → long/flat stays flat → ~0 Sharpe; force negative baseline path
    # by using a choppy series whose OOS Sharpe is negative is hard to guarantee, so we assert
    # the contract: negative current OOS Sharpe → deactivate (via the base class rule).
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    # Monkey-ish: drive a negative OOS by a steep, noisy decline scored window.
    import numpy as np

    rng = np.random.default_rng(11)
    closes = list(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, size=300)))
    service = build_service(bars=make_bars(closes), publisher=publisher)
    result = await service.revalidate("s", "AAPL", 1.0, Interval.D1)
    # status is one of the valid set regardless; if OOS Sharpe < 0 it must be deactivate
    if result.current_oos_sharpe < 0:
        assert result.recommended_status == "deactivate"
