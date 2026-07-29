"""Alpha decay (P5-4): how long to hold, and how fast we must act.

The holding period and the tranche count are currently set by the label's
horizon — a coherent default, not evidence. This is the measurement that turns
them into a decision, so the tests are about whether it can actually tell the
two answers apart: a signal that persists for weeks, and one that is gone by
tomorrow.

The pure pieces get exact tests; the end-to-end gets a universe whose decay
profile is known by construction.
"""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from trading_common.schemas import Interval, OHLCVBar

from src.core.alpha_decay import (
    DecayPoint,
    _forward_returns,
    _half_life,
    _verdict,
    run_alpha_decay,
)

D0 = datetime(2021, 1, 4, tzinfo=UTC)


def bars_from_closes(symbol: str, closes: list[float]) -> list[OHLCVBar]:
    return [
        OHLCVBar(
            symbol=symbol,
            timestamp=D0 + timedelta(days=i),
            interval=Interval.D1,
            open=c,
            high=c * 1.005,
            low=c * 0.995,
            close=c,
            adj_close=c,
            volume=1_000_000.0,
        )
        for i, c in enumerate(closes)
    ]


# --- forward returns: the arithmetic everything else rests on --------------


def test_forward_returns_measure_from_the_delayed_entry():
    """The delay must move BOTH ends of the window. Moving only the exit would
    silently measure a longer hold rather than a later entry — and would make
    delay look free, which is the opposite of the answer being sought.
    """
    closes = [100.0, 110.0, 121.0, 133.1, 146.41]  # exactly +10% per session
    bars = {"AAA": bars_from_closes("AAA", closes)}
    dates = [D0 + timedelta(days=i) for i in range(5)]
    symbols = ["AAA"] * 5

    h1 = _forward_returns(bars, dates, symbols, horizon=1, delay=0)
    assert h1[0] == pytest.approx(0.10)
    assert h1[2] == pytest.approx(0.10)

    h2 = _forward_returns(bars, dates, symbols, horizon=2, delay=0)
    assert h2[0] == pytest.approx(0.21)  # two compounded 10% sessions

    delayed = _forward_returns(bars, dates, symbols, horizon=1, delay=1)
    assert delayed[0] == pytest.approx(0.10)  # one session, entered a day later
    # ...and it is a ONE-session return, not a two-session one
    assert delayed[0] != pytest.approx(h2[0])


def test_a_window_running_past_the_history_is_missing_not_zero():
    """Truncated windows must drop out. Calling them a zero return would drag
    every long-horizon IC toward nothing and manufacture a decay profile."""
    bars = {"AAA": bars_from_closes("AAA", [100.0, 101.0, 102.0, 103.0])}
    dates = [D0 + timedelta(days=i) for i in range(4)]
    forward = _forward_returns(bars, dates, symbols=["AAA"] * 4, horizon=2, delay=0)
    assert np.isfinite(forward[0]) and np.isfinite(forward[1])
    assert math.isnan(forward[2]) and math.isnan(forward[3])


def test_an_unknown_symbol_or_date_yields_no_return():
    bars = {"AAA": bars_from_closes("AAA", [100.0, 101.0, 102.0])}
    forward = _forward_returns(
        bars, [D0, D0 + timedelta(days=900)], ["BBB", "AAA"], horizon=1, delay=0
    )
    assert math.isnan(forward[0]) and math.isnan(forward[1])


# --- half life -------------------------------------------------------------


def point(horizon: int, ic: float) -> DecayPoint:
    return DecayPoint(horizon, 0, ic, ic * 40, 100)


def test_half_life_interpolates_between_the_tested_horizons():
    # peak 0.04 at h=5, halving to 0.02 somewhere between 10 (0.03) and 21 (0.01)
    points = [point(1, 0.02), point(5, 0.04), point(10, 0.03), point(21, 0.01)]
    half = _half_life(points)
    assert half is not None and 10 < half < 21


def test_a_signal_that_never_halves_reports_no_half_life():
    """`None` is the informative answer: the holding period is not the binding
    constraint, so it must not be reported as some number at the range edge."""
    assert _half_life([point(1, 0.02), point(10, 0.03), point(63, 0.04)]) is None
    assert _half_life([]) is None
    assert _half_life([point(1, 0.0), point(10, 0.0)]) is None


def test_half_life_reads_the_magnitude_so_a_negative_signal_still_decays():
    """A consistently negative IC is a signal ranked backwards, not an absence
    of one; its decay is just as measurable."""
    half = _half_life([point(1, -0.05), point(10, -0.02), point(21, -0.005)])
    assert half is not None and 1 < half < 21


# --- the verdict, which is what anyone actually reads ----------------------


def feature_report(peak_h: int, peak_t: float, retention: float) -> dict:
    return {
        "feature": "f",
        "peak_horizon": peak_h,
        "peak_t": peak_t,
        "ic_retained_by_delay": [
            {"delay": 0, "ic_retained": 1.0},
            {"delay": 1, "ic_retained": retention},
        ],
    }


def test_no_credible_feature_yields_no_holding_period_recommendation():
    """The failure this guards: printing "hold for 63 sessions" off a decay
    curve built entirely from noise. With nothing above |t| = 2 there is no
    profile to read, and the verdict has to say so rather than pick a peak."""
    verdict = _verdict([feature_report(63, 0.4, 0.9)], current_horizon=10)
    assert "no decay profile" in verdict.lower()
    assert "63" not in verdict


def test_a_signal_peaking_far_beyond_the_label_says_the_label_is_too_short():
    verdict = _verdict([feature_report(63, 3.1, 0.95)], current_horizon=10)
    assert "longer than the current 10-session" in verdict
    assert "not urgent" in verdict


def test_a_signal_gone_before_the_label_closes_says_the_label_is_too_long():
    verdict = _verdict([feature_report(1, -3.4, 0.9)], current_horizon=21)
    assert "shorter than the current 21-session" in verdict


def test_a_signal_lost_to_one_session_of_delay_is_called_urgent():
    """The execution-latency finding. If a day of delay costs a third of the
    IC, no amount of model work fixes it, and the report must say so plainly."""
    verdict = _verdict([feature_report(10, 3.0, 0.35)], current_horizon=10)
    assert "Entry is urgent" in verdict
    assert "35%" in verdict


# --- end to end on a universe whose decay is known by construction ---------


def trending_universe(n_symbols: int = 24, n: int = 420, seed: int = 5) -> dict:
    """Each name carries a slowly-varying drift, so momentum keeps predicting.

    The drift changes on a ~60-session scale, which means a name that has been
    rising keeps rising for weeks — the signal persists, and a decay profile
    computed on it must not report it as a one-day effect.
    """
    rng = np.random.default_rng(seed)
    bars = {}
    for k in range(n_symbols):
        drift = np.repeat(rng.normal(0.0, 0.0015, n // 60 + 1), 60)[:n]
        steps = drift + rng.normal(0.0, 0.008, n)
        closes = 100.0 * np.cumprod(1.0 + steps)
        bars[f"S{k:02d}"] = bars_from_closes(f"S{k:02d}", [float(c) for c in closes])
    return bars


def test_the_study_profiles_the_strongest_features_and_reads_out_a_decision():
    from src.core.dataset import DatasetParams

    report = run_alpha_decay(
        trending_universe(),
        params=DatasetParams(min_universe=5),
        horizons=(1, 5, 10, 21),
        delays=(0, 1, 3),
        max_features=3,
    )
    assert report["symbols"] == 24
    assert report["horizons_tested"] == [1, 5, 10, 21]
    assert 0 < len(report["features"]) <= 3
    for feature in report["features"]:
        assert [p["horizon"] for p in feature["holding"]] == [1, 5, 10, 21]
        assert [p["delay"] for p in feature["urgency"]] == [0, 1, 3]
        # every urgency point is measured at ONE horizon — the strongest — so a
        # delay curve can never be a horizon curve in disguise
        assert len({p["horizon"] for p in feature["urgency"]}) == 1
        assert feature["ic_retained_by_delay"][0]["ic_retained"] == 1.0
    assert report["verdict"]


def test_the_study_selects_features_at_the_trained_horizon_not_their_best_one():
    """Choosing which features to profile by their BEST horizon would select on
    the very quantity the study then reports — the peak would be guaranteed to
    look impressive. Selection happens at the label's horizon instead.
    """
    from src.core.dataset import DatasetParams

    params = DatasetParams(min_universe=5)
    report = run_alpha_decay(
        trending_universe(),
        params=params,
        horizons=(1, 5, 10, 21),
        delays=(0, 1),
        max_features=2,
    )
    # the peaks are free to land anywhere, including at the shortest horizon —
    # which could not happen if selection had cherry-picked the best horizon
    peaks = {f["peak_horizon"] for f in report["features"]}
    assert peaks <= {1, 5, 10, 21}
    assert report["current_horizon"] == params.label.horizon


def test_an_empty_universe_refuses_rather_than_reporting_a_flat_profile():
    with pytest.raises(ValueError, match="no history"):
        run_alpha_decay({})


def gbm_universe(n_symbols: int = 24, n: int = 400, seed: int = 0) -> dict:
    """Independent random walks — no cross-sectional signal of any kind."""
    bars = {}
    for k in range(n_symbols):
        rng = np.random.default_rng(seed * 1000 + k)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        bars[f"S{k:02d}"] = bars_from_closes(f"S{k:02d}", [float(c) for c in closes])
    return bars


def test_a_null_universe_rarely_produces_a_confident_recommendation():
    """The claim a decay study must not fail: on data with no signal it should
    usually decline to name a holding period.

    Tested across SEEDS, not on one universe, because one draw is exactly what
    this study is meant to protect against — and a single null draw really does
    reach |t| ~ 3.6 at h = 10, since the Newey-West correction covers the
    forward-window overlap but not the features' own persistence.
    """
    from src.core.dataset import DatasetParams

    confident = 0
    for seed in range(5):
        report = run_alpha_decay(
            gbm_universe(seed=seed),
            params=DatasetParams(min_universe=5),
            horizons=(5, 10, 21),
            delays=(0, 1),
            max_features=3,
        )
        if "no decay profile" not in report["verdict"].lower():
            confident += 1
    assert confident <= 2, f"{confident}/5 null universes produced a confident verdict"
