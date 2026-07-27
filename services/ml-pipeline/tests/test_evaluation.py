"""Tests for OOS evaluation: AUC, Brier, top-quantile portfolio simulation."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.core.evaluation import (
    auc,
    baseline_feature_ic,
    brier,
    relative_metrics,
    selection_diagnostics,
    top_quantile_portfolio,
)

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


def test_selection_lift_is_positive_when_ranking_works():
    dates, _symbols, probs, _rets = portfolio_inputs()
    # the top-ranked name (A) is the winner on both sessions
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=float)
    diag = selection_diagnostics(dates, y, probs, quantile=0.25)
    assert diag.base_rate == pytest.approx(0.25)
    assert diag.selected_hit_rate == 1.0  # every pick was a winner
    assert diag.lift == pytest.approx(0.75)
    assert diag.pred_p10 < diag.pred_p90


def test_selection_lift_is_zero_when_ranking_is_useless():
    dates, _symbols, probs, _rets = portfolio_inputs()
    # winners are the LOWEST-ranked names → selecting the top quantile misses them
    y = np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
    diag = selection_diagnostics(dates, y, probs, quantile=0.25)
    assert diag.selected_hit_rate == 0.0
    assert diag.lift < 0  # worse than picking at random — an honest negative signal


def test_degenerate_predictions_have_no_spread():
    dates, _symbols, _probs, _rets = portfolio_inputs()
    flat = np.full(8, 0.5)
    diag = selection_diagnostics(dates, np.zeros(8), flat, quantile=0.25)
    assert diag.pred_std == 0.0  # collapsed model — the report must show it
    assert diag.pred_p10 == diag.pred_p90 == 0.5


# --- T0-5: metrics that survive a bull market ---


def bull_market_inputs(n_sessions: int = 120, n_symbols: int = 20, seed: int = 5):
    """Every name drifts up; predictions are pure noise.

    This is fold_0 of the real run in miniature: base_rate 0.68, a long-only
    book that makes money for reasons that have nothing to do with the model.
    """
    rng = np.random.default_rng(seed)
    dates, symbols, probs, rets = [], [], [], []
    for s in range(n_sessions):
        day = D0 + timedelta(days=s)
        for k in range(n_symbols):
            dates.append(day)
            symbols.append(f"S{k}")
            probs.append(float(rng.random()))  # no information whatsoever
            rets.append(float(rng.normal(0.0012, 0.01)))  # everything rises
    return dates, symbols, np.array(probs), np.array(rets)


def test_long_short_and_active_sharpe_are_insensitive_to_base_rate():
    """The audit's G2 condition, pinned: a random model in a rising market
    shows a healthy long-only Sharpe and no relative edge at all."""
    dates, symbols, probs, rets = bull_market_inputs()
    m = relative_metrics(dates, symbols, probs, rets, quantile=0.2, cost_bps=5.0)

    assert m.sharpe_benchmark_ew > 1.0  # the market itself did well
    assert abs(m.ic_mean) < 0.05  # ...and the ranking knew nothing
    assert abs(m.icir) < 0.5
    assert abs(m.sharpe_long_short) < 1.5  # both legs rose -> difference ~ 0
    assert abs(m.sharpe_active) < 1.5  # portfolio ~ benchmark


def test_relative_metrics_detect_a_real_ranking():
    """With predictions that genuinely rank forward returns, IC and the
    long-short leg must both light up."""
    rng = np.random.default_rng(7)
    dates, symbols, probs, rets = [], [], [], []
    for s in range(120):
        day = D0 + timedelta(days=s)
        for k in range(20):
            score = rng.random()
            dates.append(day)
            symbols.append(f"S{k}")
            probs.append(float(score))
            # forward return follows the score, plus noise
            rets.append(float(0.02 * (score - 0.5) + rng.normal(0, 0.004)))
    m = relative_metrics(dates, symbols, np.array(probs), np.array(rets))

    assert m.ic_mean > 0.3
    assert m.icir > 1.0
    assert m.ic_positive_share > 0.8
    assert m.sharpe_long_short > 2.0
    assert m.sharpe_active > 1.0


def test_gross_net_and_cost_drag_are_reported():
    dates, symbols, probs, rets = bull_market_inputs()
    m = relative_metrics(dates, symbols, probs, rets, cost_bps=5.0)
    assert m.sharpe_gross > m.sharpe_net  # costs can only subtract
    assert m.turnover_daily_mean > 0
    assert m.cost_drag_annualized == pytest.approx(
        m.turnover_daily_mean * 5 / 10_000 * 252, rel=1e-6
    )


def test_baseline_feature_ic_matches_a_known_ranking():
    """A feature that perfectly orders forward returns has IC ~ 1."""
    dates, rets, feature = [], [], []
    for s in range(60):
        day = D0 + timedelta(days=s)
        for k in range(10):
            dates.append(day)
            feature.append(float(k))
            rets.append(float(k) / 100.0)  # monotone in the feature
    ic = baseline_feature_ic(dates, np.array(feature), np.array(rets))
    assert ic == pytest.approx(1.0, abs=1e-9)
