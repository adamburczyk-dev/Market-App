"""Tests for position scoring — the arithmetic every backtest path shares."""

import numpy as np
import pytest

from src.core.engine import score_positions

from .conftest import trending_closes


def long_always(n: int) -> np.ndarray:
    return np.ones(n)


def flat_always(n: int) -> np.ndarray:
    return np.zeros(n)


class TestGuards:
    def test_too_few_bars_returns_empty(self):
        r = score_positions(np.array([100.0]), np.ones(1), 5.0, first_bar=1)
        assert r.n_bars == 0
        assert r.sharpe_ratio == 0.0

    def test_position_path_must_line_up_with_prices(self):
        """A mismatched path would silently score the wrong bars."""
        r = score_positions(np.array([100.0, 101.0, 102.0]), np.ones(2), 5.0, first_bar=1)
        assert r.n_bars == 0

    def test_flat_position_earns_nothing(self):
        closes = np.array(trending_closes(n=60))
        r = score_positions(closes, flat_always(60), 5.0, first_bar=1)
        assert r.total_return == 0.0
        assert r.sharpe_ratio == 0.0
        assert r.n_trades == 0


class TestScoring:
    def test_long_always_reproduces_buy_and_hold(self):
        closes = np.array(trending_closes(n=200))
        r = score_positions(closes, long_always(200), 0.0, first_bar=1)
        assert r.total_return == pytest.approx(float(closes[-1] / closes[0] - 1.0))
        assert r.n_trades == 1  # the initial entry, and nothing after

    def test_no_look_ahead_the_position_earns_the_NEXT_bar(self):
        """A position set on the last bar must earn nothing — there is no bar
        after it. A path that scored the same bar would look prescient."""
        prices = np.array([100.0, 100.0, 100.0, 200.0])
        late = np.array([0.0, 0.0, 0.0, 1.0])
        early = np.array([0.0, 0.0, 1.0, 0.0])
        assert score_positions(prices, late, 0.0, first_bar=1).total_return == 0.0
        assert score_positions(prices, early, 0.0, first_bar=1).total_return > 0.9

    def test_costs_reduce_returns_and_scale_with_turnover(self):
        closes = np.array(trending_closes(n=200))
        churn = np.array([float(i % 2) for i in range(200)])
        free = score_positions(closes, churn, 0.0, first_bar=1)
        costly = score_positions(closes, churn, 50.0, first_bar=1)
        assert costly.total_return < free.total_return
        assert costly.n_trades == free.n_trades > 50

    def test_entry_trade_is_counted(self):
        closes = np.array(trending_closes(n=60))
        position = np.concatenate([np.zeros(30), np.ones(30)])
        assert score_positions(closes, position, 10.0, first_bar=1).n_trades == 1

    def test_max_drawdown_is_non_negative(self):
        closes = np.array(trending_closes(n=200))
        r = score_positions(closes, long_always(200), 5.0, first_bar=1)
        assert r.max_drawdown >= 0.0


class TestOOSWindow:
    def test_first_bar_restricts_the_scored_window(self):
        closes = np.array(trending_closes(n=320, seed=2))
        full = score_positions(closes, long_always(320), 5.0, first_bar=1)
        tail = score_positions(closes, long_always(320), 5.0, first_bar=320 - 126)
        assert tail.n_bars == 126
        assert tail.n_bars < full.n_bars

    def test_first_bar_below_one_is_clamped_not_negative_indexed(self):
        closes = np.array(trending_closes(n=80, seed=4))
        r = score_positions(closes, long_always(80), 5.0, first_bar=0)
        assert r.n_bars == 79
