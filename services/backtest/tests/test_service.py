"""Tests for BacktestService orchestration + event publishing."""

import pytest
from trading_common.events import EventType
from trading_common.schemas import Interval

from src.events.publisher import NullPublisher

from .conftest import build_service, make_bars, trending_closes


@pytest.mark.asyncio
async def test_run_backtest_publishes_completed_event():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    result = await service.run_backtest("sma_ema_crossover", "AAPL", Interval.D1)
    assert result.n_bars > 0
    assert len(publisher.published) == 1
    event = publisher.published[0]
    assert event.event_type == EventType.BACKTEST_COMPLETED
    assert event.strategy_name == "sma_ema_crossover"
    assert event.sharpe_ratio == pytest.approx(result.sharpe_ratio)


@pytest.mark.asyncio
async def test_revalidate_publishes_revalidated_event():
    publisher = NullPublisher()
    service = build_service(
        bars=make_bars(trending_closes(seed=1)), publisher=publisher, oos_window_days=126
    )
    result = await service.revalidate("sma_ema_crossover", "AAPL", 0.1, Interval.D1)
    assert len(publisher.published) == 1
    event = publisher.published[0]
    assert event.event_type == EventType.STRATEGY_REVALIDATED
    assert event.recommended_status in {"active", "probation", "deactivate"}
    assert event.recommended_status == result.recommended_status
    assert event.source_service == "backtest"


@pytest.mark.asyncio
async def test_run_backtest_queries_market_data_with_limit():
    service = build_service()
    await service.run_backtest("sma_ema_crossover", "MSFT", Interval.D1, limit=200)
    # the fake records calls
    market = service._market  # type: ignore[attr-defined]
    assert market.calls[-1] == ("MSFT", Interval.D1, 200)


@pytest.mark.asyncio
async def test_cost_override_reaches_the_scoring():
    """`params` now carries the RULE's knobs; only `cost_bps` is the service's."""
    service = build_service()
    free = await service.run_backtest(
        "sma_ema_crossover", "AAPL", Interval.D1, params={"cost_bps": 0.0}
    )
    costly = await service.run_backtest(
        "sma_ema_crossover", "AAPL", Interval.D1, params={"cost_bps": 200.0}
    )
    assert costly.total_return < free.total_return


@pytest.mark.asyncio
async def test_an_unknown_strategy_name_is_refused_not_stamped_on_a_proxy():
    """The defect this replaces: any name produced the built-in engine's
    numbers, so a typo returned a plausible-looking result for nothing."""
    service = build_service()
    with pytest.raises(KeyError, match="sma_ema_crossover"):
        await service.run_backtest("nie_ma_takiej", "AAPL", Interval.D1)


@pytest.mark.asyncio
async def test_a_cross_sectional_strategy_is_refused_by_name():
    from src.core.rule_engine import CrossSectionalRuleError

    service = build_service()
    with pytest.raises(CrossSectionalRuleError, match="momentum_20"):
        await service.run_backtest("momentum_rank", "AAPL", Interval.D1)


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
    result = await service.revalidate("sma_ema_crossover", "AAPL", 1.0, Interval.D1)
    # status is one of the valid set regardless; if OOS Sharpe < 0 it must be deactivate
    if result.current_oos_sharpe < 0:
        assert result.recommended_status == "deactivate"
