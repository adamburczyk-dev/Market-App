"""Choosing the target: barrier width by label shape, horizon by model-free IC."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from trading_common.schemas import Interval, OHLCVBar

from src.core.labels import LabelParams, excess_barrier_label, triple_barrier_label
from src.core.target_study import (
    calibrate_barriers,
    profile_labels,
    score_targets,
)

START = datetime(2022, 1, 3, tzinfo=UTC)


def gbm(n: int, seed: int, drift: float = 0.0, vol: float = 0.015) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + rng.normal(drift, vol, size=n))


def to_bars(symbol: str, closes: np.ndarray) -> list[OHLCVBar]:
    return [
        OHLCVBar(
            symbol=symbol,
            timestamp=START + timedelta(days=i),
            interval=Interval.D1,
            open=float(c),
            high=float(c) * 1.01,
            low=float(c) * 0.99,
            close=float(c),
            volume=1e6,
            adj_close=float(c),
            source="test",
        )
        for i, c in enumerate(closes)
    ]


def universe(n_symbols: int = 25, n: int = 400) -> dict[str, list[OHLCVBar]]:
    return {f"S{k:02d}": to_bars(f"S{k:02d}", gbm(n, seed=k)) for k in range(n_symbols)}


def test_wide_barriers_degenerate_into_a_fixed_horizon_label():
    """The measurement that moved the default from 2.0 to 1.0."""
    bars = universe()
    wide = profile_labels(bars, horizon=10, pt_mult=2.0)
    narrow = profile_labels(bars, horizon=10, pt_mult=1.0)

    assert wide.vertical > 0.85, "2.0 sigma should almost never be touched"
    assert not wide.in_target_band
    assert narrow.horizontal_share > wide.horizontal_share
    assert narrow.in_target_band


def test_calibration_recommends_a_reachable_width():
    result = calibrate_barriers(universe(), horizon=10)
    assert result["recommended_mult"] <= 1.5
    band = result["target_band"]
    assert band[0] <= result["recommended_horizontal_share"] <= band[1]
    assert len(result["profiles"]) == 5
    assert "no model was fitted" in result["note"]


def test_wider_barriers_need_a_longer_horizon():
    # sigma*sqrt(h) grows with h, but so does the time available to touch it —
    # the same multiplier resolves horizontally more often over 63 sessions.
    bars = universe()
    short = profile_labels(bars, horizon=10, pt_mult=2.0)
    long = profile_labels(bars, horizon=63, pt_mult=2.0)
    assert long.horizontal_share > short.horizontal_share


def test_excess_label_measures_the_relative_move():
    """A name that tracks the market exactly has no excess move to label; the
    same name beating the market resolves upward."""
    n = 300
    market = gbm(n, seed=1, drift=0.001)
    tracker = market * 1.0
    winner = market * np.linspace(1.0, 1.5, n)  # steadily outperforms

    params = LabelParams(pt_mult=1.0, sl_mult=1.0, horizon=10, excess=True)
    flat = excess_barrier_label(tracker, market, 150, params)
    beat = excess_barrier_label(winner, market, 150, params)

    assert flat is None  # zero excess vol → nothing to measure, refused
    assert beat is not None
    assert beat.label == 1


def test_excess_and_absolute_disagree_when_the_market_moves():
    """In a falling market a name that falls less is a WINNER on the excess
    label and a loser on the absolute one — the whole point of P1-3."""
    n = 300
    market = gbm(n, seed=4, drift=-0.002, vol=0.01)
    resilient = market * np.linspace(1.0, 1.30, n)  # falls, but less than the market

    params_abs = LabelParams(pt_mult=1.0, sl_mult=1.0, horizon=10)
    params_exc = LabelParams(pt_mult=1.0, sl_mult=1.0, horizon=10, excess=True)
    i = 200
    absolute = triple_barrier_label(resilient, resilient * 1.001, resilient * 0.999, i, params_abs)
    excess = excess_barrier_label(resilient, market, i, params_exc)

    assert absolute is not None and excess is not None
    assert absolute.label == 0
    assert excess.label == 1


def test_target_scoring_is_model_free_and_ranks_candidates():
    result = score_targets(
        universe(n_symbols=22, n=320),
        horizons=(10, 21),
        multipliers=(1.0, 2.0),
        excess_options=(False, True),
        max_sessions=120,
    )
    assert result["candidates"], "no candidate produced enough labeled sessions"
    assert result["best"]["horizon"] in (10, 21)
    assert "no trials were consumed" in result["note"]
    for candidate in result["candidates"]:
        # GBM has no signal: every feature's IC must sit near zero.
        assert abs(candidate["best_feature_ic"]) < 0.2
        assert candidate["n_samples"] > 0


def test_label_params_default_to_the_calibrated_width():
    assert LabelParams().pt_mult == pytest.approx(1.0)
    assert LabelParams().sl_mult == pytest.approx(1.0)
    assert LabelParams().excess is False  # absolute stays the default until measured
