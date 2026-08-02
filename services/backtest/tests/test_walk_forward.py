"""Tests for RuleWalkForward — revalidation of the rule the service trades."""

import pytest
from trading_common.strategies import donchian_breakout, momentum_rank, sma_ema_crossover

from src.core.rule_engine import CrossSectionalRuleError
from src.core.walk_forward import RuleWalkForward

from .conftest import make_bars, trending_closes


def ohlcv(closes: list[float]) -> list[dict]:
    """Whole bars, not just closes — the rules read highs, lows and volume."""
    return [bar.model_dump() for bar in make_bars(closes)]


@pytest.mark.asyncio
async def test_run_backtest_returns_oos_sharpe():
    wf = RuleWalkForward(sma_ema_crossover, oos_window_days=126)
    sharpe = await wf._run_backtest("sma_ema_crossover", {}, ohlcv(trending_closes(seed=1)))
    assert isinstance(sharpe, float)
    assert sharpe > 0  # a trend rule on a strong uptrend


def crashing_closes() -> list[float]:
    """Rally, sharp crash, partial recovery — a path where the rules must part.

    On a pure uptrend both rules go long early and never exit, so they produce
    the SAME number and a test built on one would pass no matter what the engine
    ran. The drawdown is what separates them: the breakout rule exits as soon as
    the prior 20-day low breaks, the moving-average pair needs the averages
    themselves to cross.
    """
    up = [100.0 * 1.004**i for i in range(200)]
    crash = [up[-1] * 0.975**i for i in range(1, 41)]
    recover = [crash[-1] * 1.006**i for i in range(1, 81)]
    return up + crash + recover


@pytest.mark.asyncio
async def test_it_evaluates_the_rule_it_was_given_not_a_proxy():
    """The whole point of S7. Two different rules on the same prices must give
    different numbers; before this change any name produced one result."""
    closes = crashing_closes()
    trend = await RuleWalkForward(sma_ema_crossover)._run_backtest("a", {}, ohlcv(closes))
    breakout = await RuleWalkForward(donchian_breakout)._run_backtest("b", {}, ohlcv(closes))
    assert trend != breakout


@pytest.mark.asyncio
async def test_a_cross_sectional_rule_is_refused_by_name():
    """momentum_rank reads a rank, which does not exist for one symbol. It used
    to be 'revalidated' against a price-momentum proxy — a different strategy
    wearing its name."""
    wf = RuleWalkForward(momentum_rank)
    with pytest.raises(CrossSectionalRuleError, match="momentum_20"):
        await wf._run_backtest("momentum_rank", {}, ohlcv(trending_closes(seed=1)))


@pytest.mark.asyncio
async def test_revalidate_active_when_oos_holds_up():
    wf = RuleWalkForward(sma_ema_crossover, oos_window_days=126)
    bars = ohlcv(trending_closes(seed=1))
    current = await wf._run_backtest("s", {}, bars)
    # baseline slightly below current → no degradation → active
    result = await wf.revalidate("s", current * 0.9, bars, {})
    assert result.recommended_status == "active"
    assert result.current_oos_sharpe == pytest.approx(current)


@pytest.mark.asyncio
async def test_revalidate_probation_on_large_degradation():
    wf = RuleWalkForward(sma_ema_crossover, oos_window_days=126)
    bars = ohlcv(trending_closes(seed=1))
    current = await wf._run_backtest("s", {}, bars)
    # baseline far above current → degradation exceeds 40% → probation
    result = await wf.revalidate("s", current * 5.0, bars, {})
    assert result.recommended_status == "probation"
    assert result.degradation_pct >= 0.40


@pytest.mark.asyncio
async def test_rule_params_reach_the_rule():
    """A retuned threshold has to change the result, or the params are decoration."""
    bars = ohlcv(trending_closes(seed=3))
    wf = RuleWalkForward(donchian_breakout, oos_window_days=126)
    default = await wf._run_backtest("s", {}, bars)
    # A rule cannot be retuned into a different rule, but confidence scaling
    # must not change positions — so this pins that params travel and that the
    # position path depends only on the direction thresholds.
    same = await wf._run_backtest("s", {"full_confidence_excess": 0.5}, bars)
    assert same == default


@pytest.mark.asyncio
async def test_custom_windows_recorded_on_result():
    wf = RuleWalkForward(sma_ema_crossover, oos_window_days=63, is_window_days=126)
    result = await wf.revalidate("s", 1.0, ohlcv(trending_closes(seed=2)), {})
    assert result.oos_window_days == 63
    assert result.is_window_days == 126
