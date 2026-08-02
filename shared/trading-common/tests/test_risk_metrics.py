"""Testy wspólnej matematyki ryzyka (szereg kapitału → VaR, obsunięcie, korelacje)."""

import math

import pytest

from trading_common.risk_metrics import (
    MIN_CORRELATION_SAMPLES,
    MIN_VAR_SAMPLES,
    annualized_sharpe,
    average_pairwise_correlation,
    conditional_var,
    correlation,
    correlation_matrix,
    drawdown_series,
    historical_var,
    max_drawdown,
    returns_from_equity,
)


def test_returns_are_period_over_period():
    assert returns_from_equity([100.0, 110.0, 99.0]) == pytest.approx([0.1, -0.1])
    assert returns_from_equity([100.0]) == []
    assert returns_from_equity([]) == []


def test_a_wipeout_ends_the_series_instead_of_producing_infinity():
    """A return through zero is not a return; carrying it forward would put a
    ±inf into every statistic downstream, silently."""
    out = returns_from_equity([100.0, 50.0, 0.0, 10.0])
    assert len(out) == 2
    assert all(math.isfinite(r) for r in out)


def test_drawdown_is_measured_from_the_RUNNING_peak():
    equity = [100.0, 120.0, 90.0, 130.0, 117.0]
    dd = drawdown_series(equity)
    assert dd[0] == pytest.approx(0.0)
    assert dd[1] == pytest.approx(0.0)  # new peak
    assert dd[2] == pytest.approx(0.25)  # 90 vs peak 120
    assert dd[3] == pytest.approx(0.0)  # new peak
    assert dd[4] == pytest.approx(0.1)  # 117 vs peak 130
    assert max_drawdown(equity) == pytest.approx(0.25)


def test_drawdown_of_a_monotonic_rise_is_zero():
    assert max_drawdown([100.0 + i for i in range(50)]) == 0.0


# --- VaR ------------------------------------------------------------------


def test_var_is_an_empirical_quantile_reported_as_a_POSITIVE_loss():
    # Exactly 5 losses in 100 observations, so the 95% threshold sits on the
    # mildest of them: 5% of days are at least this bad.
    returns = [-0.05, -0.04, -0.03, -0.02, -0.01] + [0.01] * 95
    var = historical_var(returns, confidence=0.95)
    assert var is not None
    assert var > 0, "a loss must be reported positive, or a chart plots the tail upward"
    assert var == pytest.approx(0.01)


def test_var_does_not_skip_over_the_losses_it_is_meant_to_bound():
    """The floor convention lands one past the tail: with 5 losses in 100 it
    would report the first PROFIT as the 95% VaR — a loss-free VaR on a series
    that lost money on five days."""
    returns = [-0.05, -0.04, -0.03, -0.02, -0.01] + [0.01] * 95
    var = historical_var(returns, confidence=0.95)
    worst_five = sorted(returns)[:5]
    assert var is not None
    assert all(-r >= var - 1e-12 for r in worst_five)


def test_var_refuses_a_sample_too_small_to_be_a_measurement():
    """12 observations put the 95% quantile on one point — that is not a
    conservative estimate, it is a number with no sampling distribution."""
    assert historical_var([-0.01] * (MIN_VAR_SAMPLES - 1)) is None
    assert historical_var([-0.01] * MIN_VAR_SAMPLES) is not None


def test_var_rejects_a_nonsense_confidence():
    returns = [0.01] * 50
    assert historical_var(returns, confidence=0.0) is None
    assert historical_var(returns, confidence=1.0) is None


def test_cvar_is_at_least_var_because_it_averages_BEYOND_it():
    returns = [-0.20, -0.10, -0.05] + [0.01] * 97
    var = historical_var(returns, confidence=0.95)
    cvar = conditional_var(returns, confidence=0.95)
    assert var is not None and cvar is not None
    assert cvar >= var


def test_a_riskless_path_has_no_loss_quantile_but_still_reports_zero():
    returns = [0.001] * 100
    assert historical_var(returns) == 0.0  # clamped: no loss is not a negative loss


# --- Sharpe ----------------------------------------------------------------


def test_sharpe_needs_variance():
    assert annualized_sharpe([0.01] * 50) is None  # zero std → undefined, not inf
    assert annualized_sharpe([0.01]) is None
    assert annualized_sharpe([0.01, -0.01, 0.02, -0.005]) is not None


def test_sharpe_sign_follows_the_mean():
    up = annualized_sharpe([0.02, 0.01, 0.03, -0.005])
    down = annualized_sharpe([-0.02, -0.01, -0.03, 0.005])
    assert up is not None and down is not None
    assert up > 0 > down


# --- correlation -----------------------------------------------------------


def test_perfect_and_inverse_relationships():
    a = [0.01 * (i % 7) - 0.02 for i in range(40)]
    assert correlation(a, a) == pytest.approx(1.0)
    assert correlation(a, [-x for x in a]) == pytest.approx(-1.0)


def test_correlation_refuses_a_short_overlap():
    short = [0.01, -0.02] * (MIN_CORRELATION_SAMPLES // 2 - 1)
    assert correlation(short, short) is None


def test_a_constant_series_has_no_correlation_rather_than_zero():
    """Zero would claim independence was measured; there is nothing to measure."""
    varying = [0.01 * (i % 5) for i in range(40)]
    assert correlation(varying, [0.01] * 40) is None


def test_matrix_is_symmetric_with_a_unit_diagonal_and_stable_order():
    data = {
        "MSFT": [0.01 * (i % 5) for i in range(40)],
        "AAPL": [0.01 * (i % 7) for i in range(40)],
    }
    m = correlation_matrix(data)
    assert m.symbols == ["AAPL", "MSFT"]  # sorted, not dict order
    assert m.matrix[0][0] == 1.0 and m.matrix[1][1] == 1.0
    assert m.matrix[0][1] == pytest.approx(m.matrix[1][0])
    assert m.samples == 40
    assert m.coverage == 1.0


def test_coverage_distinguishes_unmeasurable_from_uncorrelated():
    """A grid of mostly-None and a grid of mostly-zero render identically as a
    heatmap and mean opposite things."""
    data = {
        "AAPL": [0.01 * (i % 5) for i in range(40)],
        "NEWCO": [0.01, -0.01, 0.02],  # listed last week: no overlap to speak of
    }
    m = correlation_matrix(data)
    assert m.matrix[0][1] is None
    assert m.coverage == 0.0
    assert m.samples == 3


def test_average_pairwise_ignores_the_diagonal():
    data = {name: [0.01 * (i % 5) for i in range(40)] for name in ("A", "B", "C")}
    m = correlation_matrix(data)
    avg = average_pairwise_correlation(m)
    assert avg == pytest.approx(1.0)  # identical series, diagonal excluded but equal anyway
    empty = correlation_matrix({"A": [0.01] * 40})
    assert average_pairwise_correlation(empty) is None


def test_the_95_percent_case_survives_binary_floating_point():
    """`(1.0 - 0.95) * 100` is 5.000000000000004, so a bare ceil returns 6 and
    the quantile lands one past the tail — on the canonical case, of all
    places. That produced VaR 0.0 on a series that lost money five times."""
    returns = [-0.05, -0.04, -0.03, -0.02, -0.01] + [0.01] * 95
    assert historical_var(returns, confidence=0.95) == pytest.approx(0.01)
    assert conditional_var(returns, confidence=0.95) == pytest.approx(0.03)
    # And the same trap at other round confidences.
    assert historical_var([-0.1] * 10 + [0.01] * 90, confidence=0.90) == pytest.approx(0.1)
