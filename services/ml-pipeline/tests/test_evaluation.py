"""Tests for OOS evaluation: AUC, Brier, top-quantile portfolio simulation."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.core.evaluation import (
    auc,
    baseline_feature_ic,
    brier,
    per_feature_ic,
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


def test_per_feature_ic_separates_a_predictor_from_noise():
    """P2-1: the instrument the stage-2 gate reads.

    One column orders the forward return, one is random. Both are scored; only
    the first may clear |t| >= 2, and the t — not the IC level — is what tells
    them apart.
    """
    rng = np.random.default_rng(7)
    dates: list[datetime] = []
    rows: list[list[float]] = []
    rets: list[float] = []
    for s in range(80):
        day = D0 + timedelta(days=s)
        for k in range(12):
            dates.append(day)
            rows.append([float(k), float(rng.normal())])
            rets.append(float(k) / 100.0 + float(rng.normal(0, 0.005)))
    table = per_feature_ic(dates, np.array(rows), ["signal", "noise"], np.array(rets))

    assert set(table) == {"signal", "noise"}
    assert table["signal"].mean > 0.9
    assert table["signal"].tstat > 2.0
    assert abs(table["noise"].tstat) < 2.0
    assert table["signal"].n_sessions == 80
    assert table["signal"].as_dict()["t"] == pytest.approx(round(table["signal"].tstat, 2))


def test_per_feature_ic_refuses_rather_than_guesses_on_one_session():
    """A single cross-section has no standard error, so it has no t."""
    dates = [D0] * 10
    rows = np.array([[float(k)] for k in range(10)])
    table = per_feature_ic(dates, rows, ["only"], np.arange(10, dtype=float))
    assert table["only"].mean == pytest.approx(1.0)
    assert table["only"].tstat == 0.0  # not "infinite confidence"


# --- T0-4: overlapping tranches ---


def test_overlapping_tranches_cut_turnover_by_the_horizon():
    """A 10-session label evaluated by a book that turns over daily is a
    different bet from the one the model was trained on. With `h` sleeves only
    1/h of capital trades per session, so turnover falls by roughly that factor.
    """
    dates, symbols, probs, rets = bull_market_inputs(n_sessions=120, n_symbols=20, seed=9)

    daily = top_quantile_portfolio(dates, symbols, probs, rets, quantile=0.2, cost_bps=5.0)
    overlapping = top_quantile_portfolio(
        dates, symbols, probs, rets, quantile=0.2, cost_bps=5.0, tranches=10
    )

    assert daily.avg_turnover > 0.5  # noise ranking -> the book churns
    assert overlapping.avg_turnover <= 1.0 / 10 + 0.02
    assert overlapping.avg_turnover < daily.avg_turnover / 5
    # holding 10 sleeves means holding far more names at once
    assert overlapping.avg_positions > daily.avg_positions


def test_overlapping_tranches_hold_positions_for_the_horizon():
    """A name selected on session t stays in the book for `tranches` sessions:
    with a ranking that changes every day, the position count converges on
    tranches x quantile x universe rather than quantile x universe."""
    dates, symbols, probs, rets = bull_market_inputs(n_sessions=60, n_symbols=20, seed=4)
    result = top_quantile_portfolio(dates, symbols, probs, rets, quantile=0.2, tranches=5)
    # 5 sleeves x 4 names, minus overlap between sleeves
    assert 8 <= result.avg_positions <= 20


def test_relative_metrics_describe_the_same_book_as_the_gate():
    """The gate scores the tranche book; the relative metrics must score THAT
    book, not a daily-rebalanced one.

    They did not, and the real run #2 printed the consequence side by side:
    "sharpe 0.79" (10-day tranches, turnover 5%) next to "sharpe_net −0.05"
    (daily, turnover 26%) for one window — two portfolios in one row.
    """
    dates, symbols, probs, rets = bull_market_inputs(n_sessions=120, n_symbols=20, seed=11)

    gate = top_quantile_portfolio(dates, symbols, probs, rets, quantile=0.2, tranches=10)
    relative = relative_metrics(dates, symbols, probs, rets, quantile=0.2, tranches=10)
    assert relative.turnover_daily_mean == pytest.approx(gate.avg_turnover, abs=1e-9)
    assert relative.sharpe_net == pytest.approx(gate.sharpe, rel=1e-6)

    # ...and the daily book really is a different object, so this matters
    daily = relative_metrics(dates, symbols, probs, rets, quantile=0.2)
    assert daily.turnover_daily_mean > 5 * relative.turnover_daily_mean


def test_relative_metrics_default_to_the_daily_book():
    # tranches=1 must reproduce the previously pinned behaviour exactly.
    dates, symbols, probs, rets = portfolio_inputs()
    a = relative_metrics(dates, symbols, probs, rets, quantile=0.25, cost_bps=10.0)
    b = relative_metrics(dates, symbols, probs, rets, quantile=0.25, cost_bps=10.0, tranches=1)
    assert a == b


def test_single_tranche_is_the_previous_behaviour():
    dates, symbols, probs, rets = portfolio_inputs()
    a = top_quantile_portfolio(dates, symbols, probs, rets, quantile=0.25, cost_bps=10.0)
    b = top_quantile_portfolio(
        dates, symbols, probs, rets, quantile=0.25, cost_bps=10.0, tranches=1
    )
    assert a == b


# --- per-symbol costs (P5-2) ------------------------------------------------


def test_a_flat_cost_table_reproduces_the_flat_rate_exactly():
    """The property that makes per-name costs a safe addition rather than a
    redefinition: give every name the same cost and nothing may move. Without
    this, every historical result silently becomes incomparable to a new one.
    """
    dates, symbols, probs, rets = bull_market_inputs(n_sessions=120, n_symbols=20, seed=7)
    flat_table = dict.fromkeys(set(symbols), 8.0)
    for tranches in (1, 10):
        base = top_quantile_portfolio(dates, symbols, probs, rets, cost_bps=8.0, tranches=tranches)
        table = top_quantile_portfolio(
            dates,
            symbols,
            probs,
            rets,
            cost_bps=8.0,
            tranches=tranches,
            cost_bps_by_symbol=flat_table,
        )
        assert base == table, f"tranches={tranches}"
        assert relative_metrics(
            dates, symbols, probs, rets, cost_bps=8.0, tranches=tranches
        ) == relative_metrics(
            dates,
            symbols,
            probs,
            rets,
            cost_bps=8.0,
            tranches=tranches,
            cost_bps_by_symbol=flat_table,
        )


def test_the_book_pays_for_the_names_it_actually_buys():
    """A flat rate charges the average name; a real book charges the names the
    model picked. If the model happens to like the expensive half of the
    universe, the flat rate flatters it — and that is exactly the failure a
    per-name table exists to expose.
    """
    # Half the universe is always preferred, and which of those it holds still
    # rotates — so the book demonstrably buys one group, and buys it often.
    rng = np.random.default_rng(9)
    dates, symbols, probs, rets = [], [], [], []
    for s in range(120):
        day = D0 + timedelta(days=s)
        for k in range(20):
            dates.append(day)
            symbols.append(f"S{k}")
            probs.append(float(rng.random()) + (1.0 if k < 10 else 0.0))
            rets.append(float(rng.normal(0.0012, 0.01)))
    probs, rets = np.array(probs), np.array(rets)

    favourites = {f"S{k}" for k in range(10)}
    rest = {f"S{k}" for k in range(10, 20)}
    expensive = {**dict.fromkeys(favourites, 50.0), **dict.fromkeys(rest, 5.0)}
    cheap = {**dict.fromkeys(favourites, 5.0), **dict.fromkeys(rest, 50.0)}

    dear = relative_metrics(dates, symbols, probs, rets, cost_bps=5.0, cost_bps_by_symbol=expensive)
    keen = relative_metrics(dates, symbols, probs, rets, cost_bps=5.0, cost_bps_by_symbol=cheap)
    assert dear.cost_drag_annualized > keen.cost_drag_annualized
    assert dear.sharpe_net < keen.sharpe_net
    # gross is untouched — costs must not leak into the pre-cost series
    assert dear.sharpe_gross == pytest.approx(keen.sharpe_gross)
    # ...and turnover is a property of the book, not of what it paid
    assert dear.turnover_daily_mean == pytest.approx(keen.turnover_daily_mean)


def test_an_uncosted_name_falls_back_to_the_flat_rate_not_to_free():
    """A partial table is the normal case — a name can drop out of the cost run
    for want of history. Treating the gap as zero would make the least-known
    names look like the cheapest ones to trade."""
    dates, symbols, probs, rets = bull_market_inputs(n_sessions=60, n_symbols=10, seed=3)
    full = relative_metrics(dates, symbols, probs, rets, cost_bps=20.0)
    empty_table = relative_metrics(
        dates, symbols, probs, rets, cost_bps=20.0, cost_bps_by_symbol={}
    )
    assert full == empty_table


# --- overlapping-window t-statistics (found via P5-4) ----------------------


def test_the_default_ic_tstat_is_unchanged_for_one_session_returns():
    """Every existing caller scores `next_returns` — one session, no overlap —
    so the correction must be strictly opt-in or it silently restates the whole
    E2 feature table and the sector study."""
    dates, symbols, probs, rets = bull_market_inputs(n_sessions=120, n_symbols=20, seed=2)
    del symbols
    x = np.column_stack([probs, probs**2])
    a = per_feature_ic(dates, x, ["a", "b"], rets)
    b = per_feature_ic(dates, x, ["a", "b"], rets, overlap=1)
    assert a == b


def test_overlapping_forward_windows_do_not_get_an_independent_sample_tstat():
    """The bug P5-4's live check caught: consecutive ICs measured on windows
    that share h-1 sessions are not independent draws, so dividing by sqrt(n)
    understates the error by roughly sqrt(h). It inflates MORE at longer
    horizons, which is what makes it dangerous for a decay profile — the
    holding-period recommendation would drift to the longest horizon tested for
    a purely mechanical reason.

    Overlapping returns are only half of it — and getting this wrong is easy.
    If the feature were serially independent, consecutive ICs would be nearly
    independent too despite the shared return window. The autocorrelation comes
    from the feature ALSO being persistent, which every real one is: momentum,
    RSI and distance-to-average all change slowly. So the feature here is a
    trailing sum, like momentum, against an overlapping forward sum.
    """
    rng = np.random.default_rng(11)
    n_sessions, n_symbols, horizon = 200, 12, 10
    steps = rng.normal(0.0, 0.01, (n_sessions + 2 * horizon, n_symbols))
    dates, feature, forward = [], [], []
    for s in range(horizon, n_sessions + horizon):
        for k in range(n_symbols):
            dates.append(D0 + timedelta(days=s))
            feature.append(float(steps[s - horizon : s, k].sum()))  # trailing momentum
            forward.append(float(steps[s + 1 : s + 1 + horizon, k].sum()))
    x = np.asarray(feature).reshape(-1, 1)
    rets = np.asarray(forward)

    naive = per_feature_ic(dates, x, ["f"], rets)["f"]
    corrected = per_feature_ic(dates, x, ["f"], rets, overlap=horizon)["f"]
    assert corrected.mean == naive.mean  # only the error bar changes
    assert corrected.n_sessions == naive.n_sessions
    assert abs(corrected.tstat) < abs(naive.tstat), (naive.tstat, corrected.tstat)


def test_the_correction_is_neutral_when_the_series_is_not_autocorrelated():
    """It must not be a blanket penalty. Given genuinely independent ICs, the
    corrected error should land close to the plain one — otherwise it would just
    be a thumb on the scale against long horizons."""
    rng = np.random.default_rng(5)
    n_sessions, n_symbols = 300, 15
    dates, feature, forward = [], [], []
    for s in range(n_sessions):
        for _k in range(n_symbols):
            dates.append(D0 + timedelta(days=s))
            feature.append(float(rng.normal()))
            forward.append(float(rng.normal()))  # independent across sessions
    x = np.asarray(feature).reshape(-1, 1)
    rets = np.asarray(forward)
    naive = per_feature_ic(dates, x, ["f"], rets)["f"]
    corrected = per_feature_ic(dates, x, ["f"], rets, overlap=10)["f"]
    assert abs(corrected.tstat) == pytest.approx(abs(naive.tstat), rel=0.5)
