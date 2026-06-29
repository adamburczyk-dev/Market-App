"""Tests for EngineWalkForward — the concrete backtest-backed revalidation."""

import pytest

from src.core.engine import BacktestParams
from src.core.walk_forward import EngineWalkForward

from .conftest import trending_closes


def ohlcv(closes: list[float]) -> list[dict]:
    return [{"close": c} for c in closes]


@pytest.mark.asyncio
async def test_run_backtest_returns_oos_sharpe():
    wf = EngineWalkForward(BacktestParams(lookback=20), oos_window_days=126)
    sharpe = await wf._run_backtest("momentum_rank", {}, ohlcv(trending_closes(seed=1)))
    assert isinstance(sharpe, float)
    assert sharpe > 0  # uptrend → positive OOS Sharpe


@pytest.mark.asyncio
async def test_revalidate_active_when_oos_holds_up():
    wf = EngineWalkForward(BacktestParams(lookback=20), oos_window_days=126)
    closes = trending_closes(seed=1)
    current = await wf._run_backtest("s", {}, ohlcv(closes))
    # baseline slightly below current → no degradation → active
    result = await wf.revalidate("s", current * 0.9, ohlcv(closes), {})
    assert result.recommended_status == "active"
    assert result.current_oos_sharpe == pytest.approx(current)


@pytest.mark.asyncio
async def test_revalidate_probation_on_large_degradation():
    wf = EngineWalkForward(BacktestParams(lookback=20), oos_window_days=126)
    closes = trending_closes(seed=1)
    current = await wf._run_backtest("s", {}, ohlcv(closes))
    # baseline far above current → degradation exceeds 40% → probation
    result = await wf.revalidate("s", current * 5.0, ohlcv(closes), {})
    assert result.recommended_status == "probation"
    assert result.degradation_pct >= 0.40


@pytest.mark.asyncio
async def test_params_override_passed_through():
    wf = EngineWalkForward(BacktestParams(lookback=20))
    merged = wf._merge_params({"lookback": 5, "cost_bps": 12.0})
    assert merged.lookback == 5
    assert merged.cost_bps == 12.0
    assert merged.entry_momentum == 0.0  # untouched default


@pytest.mark.asyncio
async def test_custom_windows_recorded_on_result():
    wf = EngineWalkForward(oos_window_days=63, is_window_days=126)
    result = await wf.revalidate("s", 1.0, ohlcv(trending_closes(seed=2)), {})
    assert result.oos_window_days == 63
    assert result.is_window_days == 126
