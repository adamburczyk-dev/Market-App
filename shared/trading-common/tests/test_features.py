"""Testy liczenia FeatureVector z barów OHLCV (shared feature definitions)."""

import math
from datetime import UTC, datetime, timedelta

import pytest

from trading_common.features import (
    FEATURE_LOOKBACK,
    FULL_HISTORY,
    RULE_ONLY_FEATURES,
    TECHNICAL_FEATURES,
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


# --- S4: the classic-TA block (rule strategies only) ----------------------


def ohlc_bars(
    closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None
) -> list[OHLCVBar]:
    """Bars with explicit highs/lows — needed to assert ATR and Donchian."""
    base = datetime(2020, 1, 1, tzinfo=UTC)
    hs = highs if highs is not None else [c + 1.0 for c in closes]
    ls = lows if lows is not None else [c - 1.0 for c in closes]
    return [
        OHLCVBar(
            symbol="TEST",
            timestamp=base + timedelta(days=i),
            interval=Interval.D1,
            open=c,
            high=h,
            low=lo,
            close=c,
            volume=1_000_000.0,
            source="test",
        )
        for i, (c, h, lo) in enumerate(zip(closes, hs, ls, strict=True))
    ]


def test_declared_technical_features_match_what_is_produced():
    """TECHNICAL_FEATURES is hand-written; this is what stops it drifting.

    The registry validates rule inputs against that set, so a feature added
    without declaring it would make a legitimate rule un-registrable — and a
    name declared without being produced would let a rule register and then
    HOLD forever.
    """
    produced = set(compute_feature_vector(make_bars(FULL_HISTORY)).features)
    assert produced == set(TECHNICAL_FEATURES)


def test_rule_only_features_are_all_produced_and_none_are_model_inputs():
    produced = set(compute_feature_vector(make_bars(FULL_HISTORY)).features)
    assert produced >= RULE_ONLY_FEATURES
    # The classic-TA block must not silently join the model's feature contract.
    assert TECHNICAL_FEATURES >= RULE_ONLY_FEATURES


def test_classic_indicators_are_degenerate_on_a_flat_series():
    """Reference values that need no arithmetic: on a constant series every
    trend indicator is zero and the bands collapse."""
    fv = compute_feature_vector(ohlc_bars([100.0] * 60))
    assert fv.features["ema_12"] == pytest.approx(100.0)
    assert fv.features["ema_26"] == pytest.approx(100.0)
    assert fv.features["macd"] == pytest.approx(0.0)
    assert fv.features["macd_hist"] == pytest.approx(0.0)
    assert fv.features["bb_width"] == pytest.approx(0.0)
    # Zero-width band → %B is undefined and must be ABSENT, not 0.5: a
    # made-up band position would put a reversion rule at a fake extreme.
    assert "bb_pct_b" not in fv.features
    # high/low are ±1 around a flat close, so the true range is exactly 2.
    assert fv.features["atr_14"] == pytest.approx(2.0)
    assert fv.features["atr_pct_14"] == pytest.approx(0.02)


def test_ema_matches_the_textbook_recursion():
    closes = [100.0 + i for i in range(60)]
    fv = compute_feature_vector(ohlc_bars(closes))

    def reference(period: int) -> float:
        alpha = 2.0 / (period + 1.0)
        ema = sum(closes[:period]) / period
        for value in closes[period:]:
            ema = alpha * value + (1.0 - alpha) * ema
        return ema

    assert fv.features["ema_12"] == pytest.approx(reference(12))
    assert fv.features["ema_26"] == pytest.approx(reference(26))
    # In a rising series the faster EMA leads, so MACD is positive.
    assert fv.features["macd"] == pytest.approx(reference(12) - reference(26))
    assert fv.features["macd"] > 0


def test_macd_signal_is_an_ema_of_the_macd_series_not_of_price():
    """The signal line starts where MACD does; feeding it the leading NaNs
    would poison the seed and drop the whole family without an error."""
    fv = compute_feature_vector(ohlc_bars([100.0 + i for i in range(60)]))
    assert "macd_signal" in fv.features
    assert fv.features["macd_hist"] == pytest.approx(
        fv.features["macd"] - fv.features["macd_signal"]
    )
    # 26 bars give a MACD value but not yet the 9 needed for its EMA.
    short = compute_feature_vector(ohlc_bars([100.0 + i for i in range(30)]))
    assert "macd" in short.features
    assert "macd_signal" not in short.features


def test_bollinger_percent_b_locates_the_close_in_the_band():
    closes = [100.0, 102.0] * 30
    fv = compute_feature_vector(ohlc_bars(closes))
    middle = sum(closes[-20:]) / 20
    sd = (sum((c - middle) ** 2 for c in closes[-20:]) / 20) ** 0.5
    upper, lower = middle + 2 * sd, middle - 2 * sd
    assert fv.features["bb_upper"] == pytest.approx(upper)
    assert fv.features["bb_pct_b"] == pytest.approx((closes[-1] - lower) / (upper - lower))
    assert fv.features["bb_width"] == pytest.approx((upper - lower) / middle)


def test_donchian_excludes_todays_bar_so_a_breakout_can_happen():
    """A channel containing today's own high can never be broken — the rule
    keyed on it would be silent forever without ever erroring."""
    closes = [100.0] * 30 + [110.0]
    highs = [100.5] * 30 + [110.5]
    lows = [99.5] * 30 + [109.5]
    fv = compute_feature_vector(ohlc_bars(closes, highs, lows))
    assert fv.features["donchian_high_20"] == pytest.approx(100.5)
    # 110 is above the prior 20-day high → position > 1.0 = breakout.
    assert fv.features["donchian_pos_20"] > 1.0
    # And a break below the prior low goes negative.
    down = compute_feature_vector(
        ohlc_bars([100.0] * 30 + [90.0], [100.5] * 30 + [90.5], [99.5] * 30 + [89.5])
    )
    assert down.features["donchian_pos_20"] < 0.0


def test_atr_counts_the_overnight_gap():
    """The reason ATR beats high-minus-low for stops: a gap IS range."""
    flat = compute_feature_vector(ohlc_bars([100.0] * 40))
    # Same intraday range, but each bar opens 3 above the previous close.
    gapped = compute_feature_vector(ohlc_bars([100.0 + 3.0 * i for i in range(40)]))
    assert flat.features["atr_14"] == pytest.approx(2.0)
    assert gapped.features["atr_14"] > flat.features["atr_14"]


def test_indicators_use_the_adjusted_scale():
    """All of them must move together with the adjustment, or a rule comparing
    two of them would compare prices from two different scales."""
    closes = [100.0 + i for i in range(60)]
    bars = ohlc_bars(closes)
    halved = [b.model_copy(update={"adj_close": b.close / 2.0}) for b in bars]
    plain = compute_feature_vector(bars)
    adjusted = compute_feature_vector(halved)
    assert adjusted.features["ema_12"] == pytest.approx(plain.features["ema_12"] / 2.0)
    assert adjusted.features["atr_14"] == pytest.approx(plain.features["atr_14"] / 2.0)
    # ...while the scale-free forms are invariant, which is what makes them
    # safe to apply to the RAW execution price.
    assert adjusted.features["atr_pct_14"] == pytest.approx(plain.features["atr_pct_14"])
    assert adjusted.features["bb_pct_b"] == pytest.approx(plain.features["bb_pct_b"])
    assert adjusted.features["close"] == pytest.approx(plain.features["close"])


def test_the_classic_block_does_not_raise_the_required_window():
    """MACD warms up in ~35 sessions, well inside the year momentum_12_1 needs.
    If a future indicator changes that, FULL_HISTORY has to move with it."""
    assert compute_feature_vector(make_bars(FULL_HISTORY)).features.keys() >= RULE_ONLY_FEATURES
    assert FULL_HISTORY == 253


# --- Phase-1 checklist families, added as CANDIDATES -----------------------


def test_stochastic_places_the_close_in_the_high_low_range():
    """Reference values with no arithmetic: a close at the window high is 100,
    at the low is 0."""
    closes = [100.0] * 19 + [110.0]
    highs = [110.0] * 20
    lows = [100.0] * 20
    at_high = compute_feature_vector(ohlc_bars(closes, highs, lows))
    assert at_high.features["stoch_k_14"] == pytest.approx(100.0)

    at_low = compute_feature_vector(ohlc_bars([100.0] * 20, highs, lows))
    assert at_low.features["stoch_k_14"] == pytest.approx(0.0)


def test_stochastic_d_is_the_average_of_three_k_readings():
    closes = [100.0 + i for i in range(30)]
    fv = compute_feature_vector(ohlc_bars(closes))
    # In a monotonic rise every %K is at the top of its own window, so %D
    # equals %K — a degenerate case, but it pins the averaging shape.
    assert fv.features["stoch_d_14"] == pytest.approx(fv.features["stoch_k_14"], abs=1e-6)


def test_stochastic_is_absent_on_a_flat_range_rather_than_dividing_by_zero():
    fv = compute_feature_vector(ohlc_bars([100.0] * 30, [100.0] * 30, [100.0] * 30))
    assert "stoch_k_14" not in fv.features


def test_cci_uses_MEAN_absolute_deviation_not_standard_deviation():
    """Lambert's definition. Using std would rescale the ±100 convention and
    make every published threshold mean something different."""
    closes = [100.0 + (5.0 if i % 2 else -5.0) for i in range(30)]
    bars = ohlc_bars(closes, closes, closes)  # typical price == close
    fv = compute_feature_vector(bars)

    window = closes[-20:]
    mean = sum(window) / 20
    mad = sum(abs(c - mean) for c in window) / 20
    expected = (window[-1] - mean) / (0.015 * mad)
    assert fv.features["cci_20"] == pytest.approx(expected)


def test_mfi_is_100_when_every_move_is_up_and_0_when_every_move_is_down():
    rising = compute_feature_vector(ohlc_bars([100.0 + i for i in range(30)]))
    falling = compute_feature_vector(ohlc_bars([200.0 - i for i in range(30)]))
    assert rising.features["mfi_14"] == pytest.approx(100.0)
    assert falling.features["mfi_14"] == pytest.approx(0.0)


def test_mfi_differs_from_rsi_because_volume_counts():
    """If they agreed the family would be a duplicate — the whole reason to add
    MFI is that it weighs each move by the money behind it."""
    closes = [100.0 + (3.0 if i % 2 else -2.0) for i in range(40)]
    heavy_up = [2_000_000.0 if i % 2 else 500_000.0 for i in range(40)]
    plain = compute_feature_vector(ohlc_bars(closes))
    weighted = compute_feature_vector(
        [
            bar.model_copy(update={"volume": volume})
            for bar, volume in zip(ohlc_bars(closes), heavy_up, strict=True)
        ]
    )
    assert weighted.features["mfi_14"] != pytest.approx(plain.features["rsi_14"])


def test_vwap_ratio_is_one_when_price_never_moves():
    fv = compute_feature_vector(ohlc_bars([100.0] * 30, [100.0] * 30, [100.0] * 30))
    assert fv.features["vwap_ratio_20"] == pytest.approx(1.0)


def test_vwap_ratio_is_above_one_in_a_rise():
    fv = compute_feature_vector(ohlc_bars([100.0 + i for i in range(30)]))
    assert fv.features["vwap_ratio_20"] > 1.0


def test_obv_and_ad_are_stored_as_SLOPES_not_levels():
    """A cumulative sum's cross-sectional rank ranks how long a symbol has been
    listed. Only the slope carries information, so only the slope is kept."""
    fv = compute_feature_vector(ohlc_bars([100.0 + i for i in range(40)]))
    assert "obv_slope_20" in fv.features and "obv" not in fv.features
    assert "ad_slope_20" in fv.features and "ad_line" not in fv.features
    assert fv.features["obv_slope_20"] > 0  # every session closes up


def test_obv_slope_is_normalized_by_volume_so_it_compares_across_sizes():
    closes = [100.0 + i for i in range(40)]
    small = compute_feature_vector(ohlc_bars(closes))
    big = compute_feature_vector(
        [b.model_copy(update={"volume": b.volume * 1000}) for b in ohlc_bars(closes)]
    )
    assert big.features["obv_slope_20"] == pytest.approx(small.features["obv_slope_20"], rel=1e-6)


def test_aroon_is_100_when_the_extreme_is_today():
    rising = compute_feature_vector(ohlc_bars([100.0 + i for i in range(40)]))
    assert rising.features["aroon_up_25"] == pytest.approx(100.0)
    assert rising.features["aroon_osc_25"] > 0

    falling = compute_feature_vector(ohlc_bars([200.0 - i for i in range(40)]))
    assert falling.features["aroon_down_25"] == pytest.approx(100.0)
    assert falling.features["aroon_osc_25"] < 0


def test_adx_is_direction_agnostic_but_di_is_not():
    """ADX says how strongly a name trends, not which way — which is exactly
    why a breakout rule and a reversion rule want opposite readings of it."""
    up = compute_feature_vector(ohlc_bars([100.0 * 1.01**i for i in range(80)]))
    down = compute_feature_vector(ohlc_bars([100.0 * 0.99**i for i in range(80)]))

    assert up.features["adx_14"] == pytest.approx(down.features["adx_14"], rel=0.25)
    assert up.features["plus_di_14"] > up.features["minus_di_14"]
    assert down.features["minus_di_14"] > down.features["plus_di_14"]


def test_adx_is_higher_in_a_trend_than_in_a_range():
    trending = compute_feature_vector(ohlc_bars([100.0 * 1.01**i for i in range(80)]))
    ranging = compute_feature_vector(
        ohlc_bars([100.0 + (2.0 if i % 2 else -2.0) for i in range(80)])
    )
    assert trending.features["adx_14"] > ranging.features["adx_14"]


def test_keltner_position_is_zero_at_the_middle_band():
    fv = compute_feature_vector(ohlc_bars([100.0] * 40))
    assert fv.features["keltner_pos_20"] == pytest.approx(0.0)


def test_every_new_family_is_a_CANDIDATE_not_a_model_input():
    """The point of the stage: they are computed and measurable, not adopted."""
    added = {
        "stoch_k_14",
        "stoch_d_14",
        "cci_20",
        "mfi_14",
        "vwap_ratio_20",
        "obv_slope_20",
        "ad_slope_20",
        "aroon_up_25",
        "aroon_down_25",
        "aroon_osc_25",
        "plus_di_14",
        "minus_di_14",
        "adx_14",
        "keltner_pos_20",
    }
    assert added <= RULE_ONLY_FEATURES
    assert added <= TECHNICAL_FEATURES


def test_short_series_do_not_crash_the_new_families():
    """A negative slice start indexes from the END of the array in Python, so
    "not enough history" became an empty window and a ValueError three
    functions away. Every length from 1 up must yield a vector."""
    for n in range(1, 40):
        fv = compute_feature_vector(ohlc_bars([100.0 + i * 0.5 for i in range(n)]))
        assert fv.features["close"] == pytest.approx(100.0 + (n - 1) * 0.5)


def test_stoch_d_needs_a_longer_window_than_stoch_k():
    """%D averages three %K readings, so it needs two more bars than %K — and
    must be ABSENT rather than computed from a truncated window."""
    short = compute_feature_vector(ohlc_bars([100.0 + i for i in range(15)]))
    assert "stoch_k_14" in short.features
    assert "stoch_d_14" not in short.features
    longer = compute_feature_vector(ohlc_bars([100.0 + i for i in range(17)]))
    assert "stoch_d_14" in longer.features
