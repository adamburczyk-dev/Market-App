"""Tests for triple-barrier labeling (docs/ml_integration_plan.md §4)."""

import math

import numpy as np

from src.core.labels import LabelParams, trailing_sigma, triple_barrier_label

P = LabelParams(sigma_window=20, pt_mult=2.0, sl_mult=2.0, horizon=10)


def path(closes: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OHLC path where high/low hug the close (barriers touched via close moves)."""
    c = np.array(closes, dtype=float)
    return c, c * 1.001, c * 0.999


def noisy_history(n: int = 30, base: float = 100.0, amp: float = 0.5) -> list[float]:
    """Oscillating history with non-zero vol (sigma estimable, no barrier drama)."""
    return [base + (amp if i % 2 else -amp) for i in range(n)]


def test_upper_barrier_first():
    history = noisy_history(30)
    sigma = trailing_sigma(np.array(history), 29, 20)
    jump = history[-1] * (1 + 2.5 * sigma * math.sqrt(10))  # beyond the +2σ√10 barrier
    closes, highs, lows = path(history + [jump] * 3)
    outcome = triple_barrier_label(closes, highs, lows, 29, P)
    assert outcome is not None
    assert outcome.label == 1
    assert outcome.barrier == "upper"
    assert outcome.touch_index == 30  # first bar after the sample


def test_lower_barrier_first():
    history = noisy_history(30)
    sigma = trailing_sigma(np.array(history), 29, 20)
    drop = history[-1] * (1 - 2.5 * sigma * math.sqrt(10))
    closes, highs, lows = path(history + [drop] * 3)
    outcome = triple_barrier_label(closes, highs, lows, 29, P)
    assert outcome.label == 0
    assert outcome.barrier == "lower"


def test_same_bar_double_touch_is_conservative_loss():
    history = noisy_history(30)
    sigma = trailing_sigma(np.array(history), 29, 20)
    width = 2.5 * sigma * math.sqrt(10)
    closes = np.array(history + [history[-1]] * 3)
    highs = closes * 1.001
    lows = closes * 0.999
    # one wild bar pierces BOTH barriers → lower wins
    highs[30] = history[-1] * (1 + width)
    lows[30] = history[-1] * (1 - width)
    outcome = triple_barrier_label(closes, highs, lows, 29, P)
    assert outcome.label == 0
    assert outcome.barrier == "lower"


def test_vertical_barrier_resolves_by_net_return():
    history = noisy_history(30)
    # gentle drift, far inside the ±2σ√10 barriers, full 10-bar window available
    tail_up = [history[-1] * (1 + 0.0005 * k) for k in range(1, 12)]
    closes, highs, lows = path(history + tail_up)
    outcome = triple_barrier_label(closes, highs, lows, 29, P)
    assert outcome.barrier == "vertical"
    assert outcome.label == 1
    assert outcome.touch_index == 39  # exactly horizon bars after the sample

    tail_down = [history[-1] * (1 - 0.0005 * k) for k in range(1, 12)]
    closes, highs, lows = path(history + tail_down)
    outcome = triple_barrier_label(closes, highs, lows, 29, P)
    assert outcome.barrier == "vertical"
    assert outcome.label == 0


def test_truncated_window_is_unresolved():
    history = noisy_history(30)
    closes, highs, lows = path(history + [history[-1]] * 4)  # only 4 future bars < horizon
    assert triple_barrier_label(closes, highs, lows, 29, P) is None


def test_not_enough_history_for_sigma():
    closes, highs, lows = path(noisy_history(15))  # < sigma_window
    assert triple_barrier_label(closes, highs, lows, 14, P) is None


def test_flat_history_has_no_sigma():
    closes, highs, lows = path([100.0] * 40)
    assert trailing_sigma(closes, 39, 20) is None
    assert triple_barrier_label(closes, highs, lows, 30, P) is None


def test_barrier_width_scales_with_horizon():
    history = noisy_history(30)
    sigma = trailing_sigma(np.array(history), 29, 20)
    # a move that breaches the h=5 barrier but NOT the h=10 barrier
    move = history[-1] * (1 + 2.0 * sigma * math.sqrt(7))
    closes, highs, lows = path(history + [move] * 12)
    short = triple_barrier_label(closes, highs, lows, 29, LabelParams(horizon=5))
    long = triple_barrier_label(closes, highs, lows, 29, LabelParams(horizon=10))
    assert short.barrier == "upper"  # √5-scaled barrier is tighter → touched
    assert long.barrier == "vertical"  # √10-scaled barrier holds → time decides
