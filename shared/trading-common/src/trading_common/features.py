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
"""

import math

import numpy as np

from trading_common.schemas import FeatureVector, OHLCVBar

_TRADING_DAYS = 252


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


def compute_feature_vector(bars: list[OHLCVBar]) -> FeatureVector:
    """Compute a Tier-1 FeatureVector from chronologically-sorted bars.

    Features are computed defensively: each one is only added when enough
    history is available, so short series still yield a (smaller) vector.
    """
    bars = sorted(bars, key=lambda b: b.timestamp)
    closes = np.array([b.close for b in bars], dtype=float)
    volumes = np.array([b.volume for b in bars], dtype=float)
    n = len(closes)
    last = bars[-1]

    feats: dict[str, float] = {"close": float(closes[-1])}

    if n >= 2:
        returns = np.diff(closes) / closes[:-1]
        feats["return_1d"] = float(returns[-1])
    if n >= 6:
        feats["return_5d"] = float(closes[-1] / closes[-6] - 1.0)
    if n >= 21:
        feats["return_20d"] = float(closes[-1] / closes[-21] - 1.0)
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

    return FeatureVector(
        symbol=last.symbol,
        timestamp=last.timestamp,
        interval=last.interval,
        features=feats,
        tier=1,
        rank_transformed=False,
    )
