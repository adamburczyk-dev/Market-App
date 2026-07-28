"""Adjusted price series — one definition of what a return means."""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trading_common.features import compute_feature_vector
from trading_common.prices import (
    adjusted_closes,
    adjusted_ohlc,
    adjustment_factors,
    has_adjusted,
)
from trading_common.schemas import Interval, OHLCVBar

START = datetime(2025, 1, 1, tzinfo=UTC)


def bar(i: int, close: float, adj: float | None = None) -> OHLCVBar:
    return OHLCVBar(
        symbol="X",
        timestamp=START + timedelta(days=i),
        interval=Interval.D1,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=1_000.0,
        adj_close=adj,
        source="test",
    )


def test_falls_back_to_raw_close_when_unadjusted():
    bars = [bar(i, 100.0 + i) for i in range(5)]
    assert not has_adjusted(bars)
    assert np.allclose(adjusted_closes(bars), [100, 101, 102, 103, 104])
    assert np.allclose(adjustment_factors(bars), 1.0)


def test_factor_applies_to_the_whole_bar():
    # A 2% dividend adjustment must move high and low too — otherwise a barrier
    # scan compares an adjusted close against unadjusted extremes.
    b = bar(0, 100.0, adj=98.0)
    o, h, low, c = adjusted_ohlc([b])
    assert c[0] == pytest.approx(98.0)
    assert h[0] == pytest.approx(101.0 * 0.98)
    assert low[0] == pytest.approx(99.0 * 0.98)
    assert o[0] == pytest.approx(98.0)


def test_dividend_gap_disappears_from_the_return():
    """The defect this exists to fix, in miniature.

    A payer holds a flat total return across an ex-dividend date: the raw close
    drops by the dividend while the adjusted series does not. Measured on raw
    prices the name looks like it lost 2%; on adjusted prices it is flat.
    """
    raw = [bar(0, 100.0, adj=98.0), bar(1, 98.0, adj=98.0)]
    adj = adjusted_closes(raw)
    raw_return = raw[1].close / raw[0].close - 1.0
    adj_return = adj[1] / adj[0] - 1.0
    assert raw_return == pytest.approx(-0.02)
    assert adj_return == pytest.approx(0.0)


def test_features_are_computed_on_the_adjusted_series():
    # 30 flat total-return sessions with a 1% dividend every 10th day: raw
    # momentum drifts negative, adjusted momentum stays at zero.
    bars, raw_price, adj_price = [], 100.0, 100.0
    for i in range(30):
        if i and i % 10 == 0:
            raw_price *= 0.99  # ex-dividend gap in the traded price
        bars.append(bar(i, raw_price, adj=adj_price))
    feats = compute_feature_vector(bars).features

    assert feats["return_20d"] == pytest.approx(0.0, abs=1e-9)
    assert feats["close"] == pytest.approx(raw_price)  # execution price stays RAW
    raw_20d = bars[-1].close / bars[-21].close - 1.0
    assert raw_20d < -0.01  # what the old code would have reported


def test_partially_adjusted_history_does_not_crash():
    bars = [bar(0, 100.0, adj=99.0), bar(1, 101.0)]
    assert not has_adjusted(bars)
    closes = adjusted_closes(bars)
    assert closes[0] == pytest.approx(99.0)
    assert closes[1] == pytest.approx(101.0)
    assert math.isfinite(float(closes.sum()))
