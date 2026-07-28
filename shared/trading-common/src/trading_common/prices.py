"""Adjusted price series — the one place that decides what a "return" means.

The raw OHLC in a bar is the price an order would have paid. It is the wrong
series to measure a return on: a dividend leaves the price on the ex-date, so a
payer's raw return understates its total return by the whole yield. Across a
cross-section that error is not noise — it is a systematic tilt against payers,
which sits exactly on the value/quality axis a ranking model is meant to read.

`adj_close` carries the split- and dividend-adjusted close. The adjustment
factor `adj_close / close` applies to the WHOLE bar: adjusting the close while
scanning raw highs and lows would compare prices from two different scales,
which matters for path-dependent labels (triple barrier).

Bars without `adj_close` (anything stored before 2026-07-28) fall back to raw
prices — the series stays usable, it just loses the dividend component.
"""

import numpy as np

from trading_common.schemas import OHLCVBar


def adjustment_factors(bars: list[OHLCVBar]) -> np.ndarray:
    """Per-bar factor `adj_close / close` (1.0 where the bar carries no adj_close)."""
    return np.array(
        [
            (bar.adj_close / bar.close) if bar.adj_close is not None and bar.close > 0 else 1.0
            for bar in bars
        ],
        dtype=float,
    )


def adjusted_closes(bars: list[OHLCVBar]) -> np.ndarray:
    """Close series to measure returns on."""
    return np.array(
        [bar.adj_close if bar.adj_close is not None else bar.close for bar in bars], dtype=float
    )


def adjusted_ohlc(bars: list[OHLCVBar]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(open, high, low, close) scaled by each bar's own adjustment factor.

    Use this wherever a decision depends on the *path* within a bar — barrier
    touches, drawdowns, gap checks — so every price compared lives on one scale.
    """
    factors = adjustment_factors(bars)
    return (
        np.array([b.open for b in bars], dtype=float) * factors,
        np.array([b.high for b in bars], dtype=float) * factors,
        np.array([b.low for b in bars], dtype=float) * factors,
        adjusted_closes(bars),
    )


def has_adjusted(bars: list[OHLCVBar]) -> bool:
    """True when every bar carries an adjusted close — i.e. returns are total returns."""
    return bool(bars) and all(b.adj_close is not None for b in bars)
