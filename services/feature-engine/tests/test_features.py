"""Testy liczenia FeatureVector z barów OHLCV."""

from trading_common.schemas import Interval

from src.core.features import compute_feature_vector

from .conftest import make_bars


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
