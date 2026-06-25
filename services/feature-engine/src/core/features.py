"""Tier-1 technical feature computation from OHLCV bars.

Pure numpy (no pandas). Reuses the framework-supplement vol_regime calculator
to derive a volatility-based exposure scalar — wiring previously-orphaned logic
into the runtime feature pipeline.
"""

import math

import numpy as np
from trading_common.schemas import FeatureVector, OHLCVBar

from src.core.calculators.vol_regime import exposure_scalar

_TRADING_DAYS = 252


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    diff = np.diff(closes[-(period + 1) :])
    gains = diff[diff > 0].sum() / period
    losses = -diff[diff < 0].sum() / period
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
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
        # vol_regime calculator expects VIX-like points (~15-40) → scale to percent.
        feats["vol_exposure_scalar"] = exposure_scalar(realized_vol * 100.0)
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
