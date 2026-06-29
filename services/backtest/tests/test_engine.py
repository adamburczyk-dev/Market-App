"""Tests for the momentum backtest engine."""

import numpy as np

from src.core.engine import BacktestParams, run_backtest

from .conftest import trending_closes


class TestGuards:
    def test_too_few_bars_returns_empty(self):
        r = run_backtest([100.0, 101.0, 102.0], BacktestParams(lookback=20))
        assert r.n_bars == 0
        assert r.sharpe_ratio == 0.0
        assert r.n_trades == 0

    def test_flat_prices_zero_sharpe(self):
        # Constant price → no momentum signal, no returns → flat, zero Sharpe.
        r = run_backtest([100.0] * 60, BacktestParams(lookback=20))
        assert r.sharpe_ratio == 0.0
        assert r.total_return == 0.0


class TestDirectionality:
    def test_uptrend_is_profitable(self):
        r = run_backtest(trending_closes(seed=1), BacktestParams(lookback=20, cost_bps=0.0))
        assert r.total_return > 0
        assert r.sharpe_ratio > 0
        assert r.n_bars > 0

    def test_downtrend_stays_flat_not_short(self):
        # Long/flat engine: a persistent downtrend → flat (no shorting) → ~0 return.
        rng = np.random.default_rng(3)
        closes = list(100.0 * np.cumprod(1.0 + rng.normal(-0.0008, 0.005, size=300)))
        r = run_backtest(closes, BacktestParams(lookback=20, cost_bps=0.0))
        # never short → can't lose more than rounding; return is ~flat, not deeply negative
        assert r.total_return > -0.05


class TestCosts:
    def test_costs_reduce_return(self):
        closes = trending_closes(seed=5)
        free = run_backtest(closes, BacktestParams(lookback=20, cost_bps=0.0))
        costly = run_backtest(closes, BacktestParams(lookback=20, cost_bps=50.0))
        assert costly.total_return < free.total_return

    def test_entry_trade_is_counted(self):
        # A clean regime flip (down then up) forces exactly one entry in the scored window.
        closes = [100.0 - i for i in range(40)] + [60.0 + 2 * i for i in range(40)]
        r = run_backtest(closes, BacktestParams(lookback=20, cost_bps=10.0))
        assert r.n_trades >= 1


class TestOOSWindow:
    def test_start_index_scores_only_tail(self):
        closes = trending_closes(seed=2, n=320)
        full = run_backtest(closes, BacktestParams(lookback=20))
        tail = run_backtest(closes, BacktestParams(lookback=20), start_index=len(closes) - 126)
        assert tail.n_bars < full.n_bars
        assert tail.n_bars <= 126

    def test_start_index_respects_warmup(self):
        # start_index below the lookback warm-up is clamped, never negative-indexed.
        closes = trending_closes(seed=4, n=80)
        r = run_backtest(closes, BacktestParams(lookback=20), start_index=0)
        assert r.n_bars > 0
        assert r.n_bars <= len(closes)
