"""Testy liczenia FeatureVector z barów OHLCV (shared feature definitions)."""

import math
from datetime import UTC, datetime, timedelta

import pytest

from trading_common.features import (
    FEATURE_LOOKBACK,
    FULL_HISTORY,
    compute_feature_vector,
)
from trading_common.schemas import Interval, OHLCVBar


def make_bars(
    n: int = 30, symbol: str = "AAPL", interval: Interval = Interval.D1
) -> list[OHLCVBar]:
    """Syntetyczne bary: lekki trend wzrostowy z oscylacją (cechy niezdegenerowane)."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    bars: list[OHLCVBar] = []
    for i in range(n):
        close = round(100 + i * 0.5 + (0.6 if i % 2 else -0.6), 2)
        bars.append(
            OHLCVBar(
                symbol=symbol,
                timestamp=base + timedelta(days=i),
                interval=interval,
                open=close - 0.5,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1_000_000.0 + i * 1000,
                source="test",
            )
        )
    return bars


def test_full_feature_vector():
    fv = compute_feature_vector(make_bars(60))
    assert fv.tier == 1
    assert fv.interval == Interval.D1
    for key in (
        "close",
        "return_1d",
        "return_5d",
        "return_20d",
        "sma_10",
        "sma_20",
        "sma_50",
        "rsi_14",
        "realized_vol_20",
        "volume_ratio",
        "momentum_20",
    ):
        assert key in fv.features, f"missing feature {key}"
    # RSI w [0, 100], realized vol dodatni
    assert 0.0 <= fv.features["rsi_14"] <= 100.0
    assert fv.features["realized_vol_20"] > 0


def test_short_series_yields_partial_vector():
    fv = compute_feature_vector(make_bars(3))
    assert "close" in fv.features
    assert "return_1d" in fv.features
    # za mało danych na te cechy
    assert "rsi_14" not in fv.features
    assert "realized_vol_20" not in fv.features
    assert "sma_50" not in fv.features


def test_timestamp_is_last_bar():
    bars = make_bars(30)
    fv = compute_feature_vector(bars)
    assert fv.timestamp == bars[-1].timestamp
    assert fv.features["close"] == bars[-1].close


# --- P2-1: the price feature family ---------------------------------------


def price_bars(closes: list[float], volumes: list[float] | None = None) -> list[OHLCVBar]:
    """Bars with exactly the closes given — so a feature can be asserted by value."""
    base = datetime(2020, 1, 1, tzinfo=UTC)
    vols = volumes if volumes is not None else [1_000_000.0] * len(closes)
    return [
        OHLCVBar(
            symbol="TEST",
            timestamp=base + timedelta(days=i),
            interval=Interval.D1,
            open=c,
            high=c * 1.01,
            low=c * 0.99,
            close=c,
            volume=v,
            source="test",
        )
        for i, (c, v) in enumerate(zip(closes, vols, strict=True))
    ]


def test_full_history_yields_every_feature():
    fv = compute_feature_vector(make_bars(FULL_HISTORY))
    for key in (
        "momentum_12_1",
        "momentum_6_1",
        "dist_52w_high",
        "max_ret_1m",
        "downside_vol_20",
        "skew_60",
        "amihud_20",
        "dollar_volume_20",
    ):
        assert key in fv.features, f"missing feature {key}"
    # ...and one bar short of it, the slowest one is genuinely absent rather
    # than silently computed from a shorter window.
    short = compute_feature_vector(make_bars(FULL_HISTORY - 1))
    assert "momentum_12_1" not in short.features
    assert "momentum_6_1" in short.features


def test_momentum_12_1_skips_the_most_recent_month():
    """The whole point of 12-1: the last 21 sessions must not count.

    A ramp that rises 1% per session for a year and then CRASHES over the last
    month has strongly positive 12-1 momentum and a strongly negative one-month
    return. If the two ever agree, the skip was lost.
    """
    closes = [100.0 * (1.01**i) for i in range(FULL_HISTORY)]
    crashed = closes[:-21] + [closes[-22] * (0.97**i) for i in range(1, 22)]
    fv = compute_feature_vector(price_bars(crashed))
    assert fv.features["momentum_12_1"] > 0.5
    assert fv.features["return_20d"] < 0.0
    # measured exactly: close[t-21] / close[t-252] - 1
    assert fv.features["momentum_12_1"] == pytest.approx(crashed[-22] / crashed[-253] - 1.0)
    assert fv.features["momentum_6_1"] == pytest.approx(crashed[-22] / crashed[-127] - 1.0)


def test_reversal_is_return_20d_and_is_not_duplicated():
    """`reversal_1m` deliberately does not exist — return_20d already is it.

    Re-adding it under a second name would recreate the momentum_20 duplication
    the dataset had to exclude: two identical columns are one vote counted twice.
    """
    fv = compute_feature_vector(make_bars(FULL_HISTORY))
    assert "reversal_1m" not in fv.features
    assert fv.features["momentum_20"] == fv.features["return_20d"]


def test_dist_52w_high_is_one_at_the_high_and_below_after_a_drop():
    rising = [100.0 + i for i in range(FULL_HISTORY)]
    at_high = compute_feature_vector(price_bars(rising)).features
    assert at_high["dist_52w_high"] == pytest.approx(1.0)
    dropped = rising[:-1] + [rising[-2] * 0.5]
    fv = compute_feature_vector(price_bars(dropped))
    assert fv.features["dist_52w_high"] == pytest.approx(dropped[-1] / max(dropped[-252:]))
    assert fv.features["dist_52w_high"] < 1.0


def test_max_ret_1m_is_the_single_best_day():
    closes = [100.0] * 40
    closes[-5] = 100.0 * 1.20  # one lottery day inside the month
    closes[-4:] = [120.0] * 4
    fv = compute_feature_vector(price_bars(closes))
    assert fv.features["max_ret_1m"] == pytest.approx(0.20)


def test_downside_vol_ignores_upside():
    """Total vol treats a violent rally as risk; semideviation must not."""
    # A series that only ever rises: volatile by the usual measure, riskless by
    # this one. That difference is the entire reason the feature exists.
    monotone = compute_feature_vector(price_bars([100.0 * 1.02**i for i in range(40)]))
    assert monotone.features["realized_vol_20"] == pytest.approx(0.0)  # constant 2% steps
    assert monotone.features["downside_vol_20"] == 0.0  # exactly: no negative days at all
    choppy = compute_feature_vector(price_bars([100.0, 110.0] * 20))
    assert 0 < choppy.features["downside_vol_20"] < choppy.features["realized_vol_20"]


def test_liquidity_features_use_raw_prices_and_scale_with_volume():
    closes = [100.0] * 40
    thin = compute_feature_vector(price_bars(closes, [1_000.0] * 40))
    thick = compute_feature_vector(price_bars(closes, [1_000_000.0] * 40))
    assert thick.features["dollar_volume_20"] == pytest.approx(100.0 * 1_000_000.0)
    # a flat series has zero |return|, so Amihud is 0 for both — move the price
    moved = [100.0 + (i % 2) for i in range(40)]
    thin_moved = compute_feature_vector(price_bars(moved, [1_000.0] * 40))
    thick_moved = compute_feature_vector(price_bars(moved, [1_000_000.0] * 40))
    assert thin_moved.features["amihud_20"] > thick_moved.features["amihud_20"]
    assert thin.features["dollar_volume_20"] < thick.features["dollar_volume_20"]


def test_skew_is_signed():
    base = [100.0] * 61
    # one large negative jump among small positive ones → left-skewed
    left = [100.0 + i * 0.01 for i in range(61)]
    left[-30] = left[-31] * 0.85
    left[-29:] = [left[-30] + i * 0.01 for i in range(1, 30)]
    fv = compute_feature_vector(price_bars(left))
    assert fv.features["skew_60"] < 0
    assert compute_feature_vector(price_bars(base)).features.get("skew_60") is None
    assert not math.isnan(fv.features["skew_60"])


def test_the_serving_window_covers_every_feature():
    """The one invariant that keeps training and serving on the same numbers.

    ml-pipeline's DatasetParams and feature-engine's FEATURE_LOOKBACK both read
    these constants, so a window that cannot produce the full feature set is a
    build-time failure here rather than a quiet neutral-0.5 fill in production.
    """
    assert FEATURE_LOOKBACK >= FULL_HISTORY
