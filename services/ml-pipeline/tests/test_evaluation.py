"""Tests for OOS evaluation: AUC, Brier, top-quantile portfolio simulation."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.core.evaluation import auc, brier, top_quantile_portfolio

D0 = datetime(2024, 6, 3, tzinfo=UTC)


def test_auc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auc_handles_ties_and_degenerate():
    y = np.array([0, 1, 0, 1])
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5  # all tied → chance
    assert auc(np.ones(4), np.array([0.1, 0.2, 0.3, 0.4])) == 0.5  # single class


def test_auc_known_value():
    # scores rank one negative above one positive → 5/6
    y = np.array([1, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.6])
    assert auc(y, s) == pytest.approx(5 / 6)


def test_brier():
    y = np.array([1.0, 0.0])
    assert brier(y, np.array([1.0, 0.0])) == 0.0
    assert brier(y, np.array([0.5, 0.5])) == pytest.approx(0.25)


def portfolio_inputs():
    """Two sessions × four symbols with hand-checkable returns."""
    dates, symbols, probs, rets = [], [], [], []
    for k, day in enumerate((D0, D0 + timedelta(days=1))):
        for j, sym in enumerate(("A", "B", "C", "D")):
            dates.append(day)
            symbols.append(sym)
            # A always ranked top, D bottom
            probs.append(0.9 - 0.2 * j)
            rets.append([0.02, 0.01, -0.01, -0.02][j] * (1 if k == 0 else 2))
    return dates, symbols, np.array(probs), np.array(rets)


def test_top_quantile_picks_best_and_charges_costs():
    dates, symbols, probs, rets = portfolio_inputs()
    result = top_quantile_portfolio(dates, symbols, probs, rets, quantile=0.25, cost_bps=10.0)
    # top-1 = A both days; day1: 2% − 10bps (initial buy), day2: 4% − 0 (no turnover)
    assert result.n_sessions == 2
    assert result.avg_positions == 1.0
    assert result.mean_daily_return == pytest.approx((0.02 - 0.001 + 0.04) / 2)
    assert result.avg_turnover == pytest.approx(0.5)  # 1.0 then 0.0
    assert result.sharpe > 0


def test_turnover_charged_on_book_changes():
    dates, symbols, probs, rets = portfolio_inputs()
    # flip the ranking on day 2 → the top name changes → full turnover both days
    flipped = probs.copy()
    flipped[4:] = flipped[4:][::-1]
    churn = top_quantile_portfolio(dates, symbols, flipped, rets, quantile=0.25, cost_bps=10.0)
    assert churn.avg_turnover == 1.0


def test_empty_inputs_yield_zero_result():
    result = top_quantile_portfolio([], [], np.array([]), np.array([]))
    assert result.n_sessions == 0
    assert result.sharpe == 0.0
