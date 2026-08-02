"""Tier-1 technical feature computation from OHLCV bars — SHARED definition.

Pure numpy (no pandas). Produces raw per-symbol features; cross-sectional
percentile ranking (López de Prado) is the separate universe-level stage in
``trading_common.ranking``.

This lives in trading-common (not feature-engine) because ml-pipeline's
training must reproduce the served features bit-for-bit over history —
duplicating the math across a service boundary guarantees train/serve skew
(docs/ml_integration_plan.md §3). feature-engine keeps orchestration, store
and API; the *definitions* are a shared contract like the event schemas.

Note: the vol_regime calculator (VIX-based) is intentionally NOT applied here —
it expects market-wide implied vol, not single-symbol realized vol. It belongs
in the macro/regime context. `realized_vol_20` is exposed as a plain feature.

P2-1 adds the documented cross-sectional families (momentum, reversal, risk,
liquidity). Two constraints shaped what could be added HERE rather than in a
universe-level stage:

- this function sees ONE symbol's bars, so anything needing a market series
  (beta, idiosyncratic vol) cannot live here without changing the serving
  contract — deferred deliberately, not forgotten;
- `momentum_12_1` and `dist_52w_high` need 252+ sessions of trailing history.
  The window is set by the CALLER (ml-pipeline's `DatasetParams.lookback` when
  training, feature-engine's `FEATURE_LOOKBACK` when serving); raising one
  without the other reintroduces train/serve skew, which is why both moved to
  300 in the same change.

There is deliberately no `reversal_1m`: the one-month return is already
`return_20d`, and adding a second name for the same column is exactly the
`momentum_20` duplication that was removed from the model input.

The classic-TA block (EMA/MACD/Bollinger/Donchian/ATR) exists for the RULE
strategies, not for the model: every name it produces is listed in
ml-pipeline's `EXCLUDED_FEATURES`. A rule needs a level to compare against;
the model gets cross-sectional ranks, where the rank of an EMA is a proxy for
share price. Whether any of these earns a place in the model input is decided
by the per-feature IC table (stage E2), never by having been computed.

Every indicator here reads the ADJUSTED series, so all of them live on one
scale and can be compared with each other. The rules consume only the
scale-free derivatives (`bb_pct_b`, `donchian_pos_20`, `atr_pct_14`, and the
sign of `macd_hist`), which is what lets them apply a result to the RAW
execution price without mixing scales.
"""

import math

import numpy as np

from trading_common.prices import adjusted_ohlc
from trading_common.schemas import FeatureVector, OHLCVBar

_TRADING_DAYS = 252
_MONTH = 21
_HALF_YEAR = 126
_YEAR = 252

# Bars required before EVERY feature is present. Set by the slowest one
# (`momentum_12_1`: a year plus the skipped month). Below this a vector is
# legal but incomplete, and the missing columns get the neutral rank 0.5.
FULL_HISTORY = _YEAR + 1

# Every technical name `compute_feature_vector` can produce given FULL_HISTORY
# bars. Declared rather than derived (deriving it would mean running the whole
# computation at import time) and pinned by a test that computes a full vector
# and asserts equality — so a feature cannot be added without landing here.
# `trading_common.strategies` validates rule inputs against this set, which is
# what turns "this rule reads a feature nobody computes" from a KeyError in
# production into a refusal at registration.
TECHNICAL_FEATURES: frozenset[str] = frozenset(
    {
        "close",
        "return_1d",
        "return_5d",
        "return_20d",
        "momentum_20",
        "sma_10",
        "sma_20",
        "sma_50",
        "price_to_sma50",
        "rsi_14",
        "realized_vol_20",
        "volume_ratio",
        "max_ret_1m",
        "downside_vol_20",
        "dollar_volume_20",
        "amihud_20",
        "skew_60",
        "momentum_6_1",
        "momentum_12_1",
        "dist_52w_high",
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_pct_b",
        "bb_width",
        "donchian_high_20",
        "donchian_low_20",
        "donchian_pos_20",
        "atr_14",
        "atr_pct_14",
        "keltner_pos_20",
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
    }
)

# Names produced for the RULE strategies only. They are price levels or band
# descriptors, so their cross-sectional rank proxies share price, not signal.
# The set lives HERE, next to the code that produces them, and ml-pipeline folds
# it into EXCLUDED_FEATURES — so an indicator added in this file cannot enter the
# model's feature contract merely by someone forgetting to exclude it there.
# A family joins the model when the per-feature IC table says so (stage E2).
RULE_ONLY_FEATURES: frozenset[str] = frozenset(
    {
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_pct_b",
        "bb_width",
        "donchian_high_20",
        "donchian_low_20",
        "donchian_pos_20",
        "atr_14",
        "atr_pct_14",
        "keltner_pos_20",
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
    }
)

# The trailing window both paths feed this function. It lives here — next to the
# definitions it constrains — because training (ml-pipeline `DatasetParams`) and
# serving (feature-engine `FEATURE_LOOKBACK`) must pass the SAME window or they
# compute different numbers under identical feature names. Two independent
# defaults are a skew waiting to happen; one shared constant is not.
FEATURE_LOOKBACK = 300


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """RSI with Wilder's smoothing over all available history (textbook RSI)."""
    deltas = np.diff(closes)
    if len(deltas) < period:
        return 50.0
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Seed with a simple average of the first `period`, then Wilder-smooth the rest.
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA series aligned with `values`; the first `period - 1` entries are NaN.

    Seeded with the SMA of the first `period` values (the textbook seed) rather
    than with values[0], which would leave the level dependent on how much
    history the caller happened to pass — the same skew `FEATURE_LOOKBACK`
    exists to prevent.
    """
    out = np.full(len(values), np.nan)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    ema = float(values[:period].mean())
    out[period - 1] = ema
    for i in range(period, len(values)):
        ema = alpha * float(values[i]) + (1.0 - alpha) * ema
        out[i] = ema
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float | None:
    """Wilder's ATR over the whole series, or None when history is too short.

    True range uses the PREVIOUS close, so it counts an overnight gap as range
    — that is the whole reason ATR is preferred to high-minus-low for stops.
    """
    if len(close) < period + 1:
        return None
    prev_close = close[:-1]
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )
    atr = float(tr[:period].mean())
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + float(tr[i])) / period
    return atr


def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing — the recursion ADX, ATR and friends are defined on.

    Not the same as an EMA with alpha=1/period at the seed: Wilder seeds with a
    plain SUM of the first `period` values and then decays. Using an EMA instead
    is the usual reason a hand-rolled ADX disagrees with every chart package.
    """
    out = np.full(len(values), np.nan)
    if len(values) < period:
        return out
    total = float(values[:period].sum())
    out[period - 1] = total
    for i in range(period, len(values)):
        total = total - total / period + float(values[i])
        out[i] = total
    return out


def _directional_index(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> tuple[float, float, float] | None:
    """(+DI, -DI, ADX) — trend STRENGTH, deliberately direction-agnostic in ADX.

    ADX is what distinguishes "trending" from "ranging" without saying which
    way, which is why a breakout rule and a mean-reversion rule want opposite
    readings of the same number.
    """
    if len(close) < 2 * period + 1:
        return None
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close[:-1]
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )

    tr_s = _wilder_smooth(tr, period)
    plus_s = _wilder_smooth(plus_dm, period)
    minus_s = _wilder_smooth(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_s / tr_s
        minus_di = 100.0 * minus_s / tr_s
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx[~np.isnan(dx)]
    if len(dx) < period:
        return None
    # ADX is Wilder's average of DX; seeded with the mean of the first `period`.
    adx = float(dx[:period].mean())
    for value in dx[period:]:
        adx = (adx * (period - 1) + float(value)) / period
    return float(plus_di[-1]), float(minus_di[-1]), adx


def _slope_per_bar(values: np.ndarray) -> float:
    """Least-squares slope of `values` against bar index."""
    n = len(values)
    x = np.arange(n, dtype=float)
    x_centred = x - x.mean()
    denominator = float((x_centred**2).sum())
    if denominator == 0:
        return 0.0
    return float((x_centred * (values - values.mean())).sum() / denominator)


def compute_feature_vector(bars: list[OHLCVBar]) -> FeatureVector:
    """Compute a Tier-1 FeatureVector from chronologically-sorted bars.

    Features are computed defensively: each one is only added when enough
    history is available, so short series still yield a (smaller) vector.
    """
    bars = sorted(bars, key=lambda b: b.timestamp)
    # Returns are measured on the ADJUSTED close (dividends + splits); the raw
    # close stays the execution price and is exposed separately below. Mixing
    # the two silently biases every dividend payer's momentum downward.
    _, highs, lows, closes = adjusted_ohlc(bars)
    raw_close = float(bars[-1].close)
    # Dollar volume is the amount of money that actually changed hands, so it
    # is the RAW close times the raw volume — adjusting the price here would
    # inflate historical turnover by every split factor since.
    raw_closes = np.array([b.close for b in bars], dtype=float)
    volumes = np.array([b.volume for b in bars], dtype=float)
    n = len(closes)
    last = bars[-1]

    feats: dict[str, float] = {"close": raw_close}

    if n >= 2:
        returns = np.diff(closes) / closes[:-1]
        feats["return_1d"] = float(returns[-1])
    if n >= 6:
        feats["return_5d"] = float(closes[-1] / closes[-6] - 1.0)
    if n >= 21:
        feats["return_20d"] = float(closes[-1] / closes[-21] - 1.0)
        # DEPRECATED alias of return_20d, kept only because the strategy rule still
        # reads this name (T1-4 replaces it with momentum_12_1). It is perfectly
        # collinear with return_20d, so the ML dataset EXCLUDES it — two identical
        # columns are a duplicated vote, not two features.
        feats["momentum_20"] = feats["return_20d"]

    for window in (10, 20, 50):
        if n >= window:
            feats[f"sma_{window}"] = float(closes[-window:].mean())
    if n >= 50:
        sma50 = closes[-50:].mean()
        if sma50 > 0:
            feats["price_to_sma50"] = float(closes[-1] / sma50)

    if n >= 15:
        feats["rsi_14"] = _rsi(closes, 14)

    if n >= 21:
        daily_returns = np.diff(closes[-21:]) / closes[-21:-1]
        realized_vol = float(np.std(daily_returns, ddof=1) * math.sqrt(_TRADING_DAYS))
        feats["realized_vol_20"] = realized_vol
        avg_volume = volumes[-20:].mean()
        if avg_volume > 0:
            feats["volume_ratio"] = float(volumes[-1] / avg_volume)

        # MAX (Bali, Cakici & Whitelaw 2011): the single best day of the month.
        # Lottery-like payoffs are systematically overpriced, so this ranks
        # opposite to the mean — which is why it is not just another vol proxy.
        feats["max_ret_1m"] = float(daily_returns.max())

        # Downside semideviation against a zero target (the Sortino
        # convention): total vol treats a violent rally as risk, this does not.
        downside = np.minimum(daily_returns, 0.0)
        feats["downside_vol_20"] = float(
            math.sqrt(float(np.mean(downside**2))) * math.sqrt(_TRADING_DAYS)
        )

        # Liquidity. `dollar_volume_20` is a level, but unlike price its
        # cross-sectional rank is a real characteristic (size/liquidity), so it
        # stays in the model input while `close` and the SMAs do not.
        dollar_volume = raw_closes * volumes
        recent_dv = dollar_volume[-20:]
        mean_dv = float(recent_dv.mean())
        if mean_dv > 0:
            feats["dollar_volume_20"] = mean_dv
        # Amihud (2002) illiquidity: price impact per million dollars traded.
        # Same 20 bars as `daily_returns`, so the two arrays line up.
        tradeable = recent_dv > 0
        if bool(tradeable.any()):
            impact = np.abs(daily_returns)[tradeable] / recent_dv[tradeable]
            feats["amihud_20"] = float(np.mean(impact) * 1e6)

    if n >= 61:
        returns_60 = np.diff(closes[-61:]) / closes[-61:-1]
        sd = float(returns_60.std(ddof=1))
        if sd > 0:
            centred = (returns_60 - returns_60.mean()) / sd
            feats["skew_60"] = float(np.mean(centred**3))

    # Momentum, deliberately skipping the most recent month: the last 21
    # sessions are short-term REVERSAL (Lehmann 1990) and pool with the opposite
    # sign, which is what makes 12-1 work (Jegadeesh & Titman 1993).
    if n >= _HALF_YEAR + 1:
        feats["momentum_6_1"] = float(closes[-1 - _MONTH] / closes[-1 - _HALF_YEAR] - 1.0)
    if n >= _YEAR + 1:
        feats["momentum_12_1"] = float(closes[-1 - _MONTH] / closes[-1 - _YEAR] - 1.0)
    if n >= _YEAR:
        # George & Hwang (2004): nearness to the 52-week high predicts better
        # than the past return it is derived from. In (0, 1]; 1.0 = at the high.
        high_52w = float(closes[-_YEAR:].max())
        if high_52w > 0:
            feats["dist_52w_high"] = float(closes[-1] / high_52w)

    # --- Classic TA, for the RULE strategies only (see the module docstring).
    # Every name below is in ml-pipeline's EXCLUDED_FEATURES. ---

    if n >= 26:
        ema_12 = _ema(closes, 12)
        ema_26 = _ema(closes, 26)
        feats["ema_12"] = float(ema_12[-1])
        feats["ema_26"] = float(ema_26[-1])
        macd_line = ema_12 - ema_26
        feats["macd"] = float(macd_line[-1])
        # The signal line is an EMA *of the MACD series*, so it has to start
        # where that series does: feeding it the leading NaNs would poison the
        # SMA seed and drop the whole family without an error.
        defined = macd_line[~np.isnan(macd_line)]
        signal_line = _ema(defined, 9)
        if not math.isnan(signal_line[-1]):
            feats["macd_signal"] = float(signal_line[-1])
            feats["macd_hist"] = float(macd_line[-1] - signal_line[-1])

    if n >= 20:
        band_window = closes[-20:]
        middle = float(band_window.mean())
        # Population std is the Bollinger convention — the band describes the
        # window itself, it does not estimate a wider population.
        sd = float(band_window.std())
        upper, lower = middle + 2.0 * sd, middle - 2.0 * sd
        feats["bb_upper"] = upper
        feats["bb_lower"] = lower
        if upper > lower:
            feats["bb_pct_b"] = float((closes[-1] - lower) / (upper - lower))
        if middle > 0:
            feats["bb_width"] = float((upper - lower) / middle)

    if n >= 21:
        # Donchian channel from the PRIOR 20 bars — today's own bar is excluded
        # on purpose. A channel containing today could never be broken (its high
        # is by construction >= today's close), so the breakout rule reading
        # `donchian_pos_20 > 1.0` would never fire.
        prior_high = float(highs[-21:-1].max())
        prior_low = float(lows[-21:-1].min())
        feats["donchian_high_20"] = prior_high
        feats["donchian_low_20"] = prior_low
        if prior_high > prior_low:
            feats["donchian_pos_20"] = float((closes[-1] - prior_low) / (prior_high - prior_low))

    atr = _atr(highs, lows, closes, 14)
    if atr is not None:
        feats["atr_14"] = atr
        if closes[-1] > 0:
            # The FRACTION is what a rule may apply to the raw execution price.
            # `atr_14` itself is on the adjusted scale; multiplying a raw price
            # by an adjusted distance is the scale mix this pair avoids.
            feats["atr_pct_14"] = float(atr / closes[-1])

        # Keltner: the same idea as Bollinger with ATR instead of standard
        # deviation, so it widens on gaps rather than only on close-to-close
        # dispersion. Only the POSITION is kept — the bands themselves are
        # price levels whose cross-sectional rank is share price.
        if n >= 20:
            middle_k = float(closes[-20:].mean())
            width = 2.0 * atr
            if width > 0:
                feats["keltner_pos_20"] = float((closes[-1] - middle_k) / width)

    # --- Phase-1 checklist families, added as CANDIDATES ------------------
    # Each is in a form whose cross-sectional rank means something. That
    # constraint removed several from the checklist as written: OBV and the
    # A/D line are unbounded cumulative sums, so ranking their LEVEL ranks how
    # long a symbol has been listed. Their slope is the part that carries
    # information, and that is what is stored.

    if n >= 15:
        # Stochastic %K: where the close sits in the recent high-low range.
        # Distinct from RSI, which only ever sees closes.
        window_high = float(highs[-14:].max())
        window_low = float(lows[-14:].min())
        if window_high > window_low:
            feats["stoch_k_14"] = float(
                100.0 * (closes[-1] - window_low) / (window_high - window_low)
            )
            # %D is the 3-period average of %K; computed from the same window
            # slid back, not from a stored series.
            k_values = []
            for offset in range(3):
                end = n - offset
                start = end - 14
                # A negative start would slice from the END of the array and
                # silently hand `.max()` an empty window — Python's negative
                # indexing turns "not enough history" into a crash three
                # functions away instead of a missing feature here.
                if start < 0:
                    break
                hi = float(highs[start:end].max())
                lo = float(lows[start:end].min())
                if hi > lo:
                    k_values.append(100.0 * (closes[end - 1] - lo) / (hi - lo))
            if len(k_values) == 3:
                feats["stoch_d_14"] = float(sum(k_values) / 3.0)

    if n >= 20:
        # CCI: typical price against its own average, scaled by MEAN absolute
        # deviation (not standard deviation — that is Lambert's definition and
        # the reason the ±100 convention means what it does).
        typical = (highs[-20:] + lows[-20:] + closes[-20:]) / 3.0
        mean_typical = float(typical.mean())
        mean_deviation = float(np.abs(typical - mean_typical).mean())
        if mean_deviation > 0:
            feats["cci_20"] = float((typical[-1] - mean_typical) / (0.015 * mean_deviation))

        # VWAP as a RATIO: the level itself is a price, the ratio is not.
        volume_window = volumes[-20:]
        traded = float(volume_window.sum())
        if traded > 0:
            vwap = float((typical[-20:] * volume_window).sum() / traded)
            if vwap > 0:
                feats["vwap_ratio_20"] = float(closes[-1] / vwap)

        # OBV and A/D: slope over the window, normalized by average volume, so
        # the number is comparable across a mega-cap and a small-cap. The raw
        # cumulative level is deliberately NOT stored.
        direction = np.sign(np.diff(closes))
        obv = np.concatenate([[0.0], np.cumsum(direction * volumes[1:])])
        average_volume = float(volumes[-20:].mean())
        if average_volume > 0:
            feats["obv_slope_20"] = float(_slope_per_bar(obv[-20:]) / average_volume)
            span = highs - lows
            with np.errstate(divide="ignore", invalid="ignore"):
                multiplier = np.where(span > 0, ((closes - lows) - (highs - closes)) / span, 0.0)
            ad_line = np.cumsum(multiplier * volumes)
            feats["ad_slope_20"] = float(_slope_per_bar(ad_line[-20:]) / average_volume)

    if n >= 21:
        # Money Flow Index: RSI computed on price × volume, so it separates a
        # rally that money followed from one that it did not. Needs 21 bars, not
        # 20: each of the 20 flows is classified by its move against the PREVIOUS
        # typical price, so the window is one bar longer than it looks.
        typical_full = (highs + lows + closes) / 3.0
        money_flow = typical_full * volumes
        deltas = np.diff(typical_full[-21:])  # 20 deltas
        flows = money_flow[-20:]  # aligned one-for-one with them
        positive = float(flows[deltas > 0].sum())
        negative = float(flows[deltas < 0].sum())
        if negative > 0:
            feats["mfi_14"] = float(100.0 - 100.0 / (1.0 + positive / negative))
        elif positive > 0:
            feats["mfi_14"] = 100.0

    if n >= 26:
        # Aroon: how RECENTLY the window's extreme happened. Time-based, so it
        # says something none of the price-based families do.
        lookback = 25
        recent_high = highs[-(lookback + 1) :]
        recent_low = lows[-(lookback + 1) :]
        since_high = lookback - int(np.argmax(recent_high))
        since_low = lookback - int(np.argmin(recent_low))
        feats["aroon_up_25"] = float(100.0 * (lookback - since_high) / lookback)
        feats["aroon_down_25"] = float(100.0 * (lookback - since_low) / lookback)
        feats["aroon_osc_25"] = feats["aroon_up_25"] - feats["aroon_down_25"]

    directional = _directional_index(highs, lows, closes, 14)
    if directional is not None:
        plus_di, minus_di, adx = directional
        feats["plus_di_14"] = plus_di
        feats["minus_di_14"] = minus_di
        # ADX measures trend STRENGTH without direction — which is exactly why
        # a breakout rule and a mean-reversion rule want opposite readings of it.
        feats["adx_14"] = adx

    return FeatureVector(
        symbol=last.symbol,
        timestamp=last.timestamp,
        interval=last.interval,
        features=feats,
        tier=1,
        rank_transformed=False,
    )
