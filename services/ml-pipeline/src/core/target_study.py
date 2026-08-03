"""Choosing the target — barrier width, horizon, absolute vs excess.

Two different kinds of choice live here, and keeping them apart is the point:

**Barrier width is a property of the LABEL, not of a model.** Whether barriers
are reachable is answered by scanning real price paths and counting how labels
resolve — no model is fitted, so it costs nothing in multiple-testing terms.
The measured target is roughly half the labels touching a horizontal barrier:
at 2.0σ only 9% did, which makes the triple barrier a fixed-horizon sign label
with extra machinery.

**Horizon and absolute-vs-excess are choices about what we predict**, and those
could be made by fitting a model to each and keeping the winner — which is how
a selection bias gets built. Instead they are scored here with a MODEL-FREE
statistic: the information coefficient of each raw feature against the
candidate label. If no feature ranks a target better than chance, no model
trained on those features will either, and we learn that for free.

Every number produced here is descriptive. Nothing is fitted, nothing is
registered, and `n_trials` is untouched.
"""

from dataclasses import asdict, dataclass

import numpy as np
import structlog
from trading_common.features import FEATURE_LOOKBACK, FULL_HISTORY, compute_feature_vector
from trading_common.prices import adjusted_ohlc
from trading_common.ranking import cross_sectional_rank
from trading_common.schemas import OHLCVBar

from src.core.evaluation import spearman
from src.core.labels import (
    LABEL_HORIZON,
    LabelParams,
    excess_barrier_label,
    triple_barrier_label,
)

logger = structlog.get_logger()

DEFAULT_MULTIPLIERS = (0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT_HORIZONS = (10, 21, 63)
# The band we want: enough horizontal touches that the path matters, not so many
# that the vertical barrier never binds and the horizon stops meaning anything.
TARGET_HORIZONTAL_LOW = 0.40
TARGET_HORIZONTAL_HIGH = 0.70


@dataclass(frozen=True)
class LabelProfile:
    horizon: int
    pt_mult: float
    excess: bool
    n_labeled: int
    n_unlabeled: int
    upper: float
    lower: float
    vertical: float
    horizontal_share: float
    positive_rate: float
    in_target_band: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _market_path(closes_by_symbol: dict[str, np.ndarray], length: int) -> np.ndarray:
    """Equal-weight median index of the universe, rebased to 1.0.

    The median rather than the mean: one name doubling should not redefine
    "the market" for a 34-name cross-section.
    """
    normalized = []
    for closes in closes_by_symbol.values():
        if len(closes) == length and np.all(closes > 0):
            normalized.append(closes / closes[0])
    if not normalized:
        return np.ones(length, dtype=float)
    return np.median(np.vstack(normalized), axis=0)


def profile_labels(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    horizon: int,
    pt_mult: float,
    excess: bool = False,
    sigma_window: int = 20,
    max_samples_per_symbol: int = 400,
) -> LabelProfile:
    """Count how labels resolve for one (horizon, width, excess) combination."""
    params = LabelParams(
        sigma_window=sigma_window,
        pt_mult=pt_mult,
        sl_mult=pt_mult,
        horizon=horizon,
        excess=excess,
    )
    closes_by_symbol: dict[str, np.ndarray] = {}
    for symbol, bars in bars_by_symbol.items():
        ordered = sorted(bars, key=lambda b: b.timestamp)
        _, _, _, adj_close = adjusted_ohlc(ordered)
        closes_by_symbol[symbol] = adj_close
    lengths = {len(c) for c in closes_by_symbol.values()}
    market = _market_path(closes_by_symbol, max(lengths)) if lengths else np.ones(1)

    counts = {"upper": 0, "lower": 0, "vertical": 0}
    unlabeled = 0
    positives = 0
    for bars in bars_by_symbol.values():
        ordered = sorted(bars, key=lambda b: b.timestamp)
        _, highs, lows, closes = adjusted_ohlc(ordered)
        n = len(closes)
        start = max(sigma_window, n - max_samples_per_symbol)
        for i in range(start, n):
            if excess:
                outcome = (
                    excess_barrier_label(closes, market, i, params) if len(market) == n else None
                )
            else:
                outcome = triple_barrier_label(closes, highs, lows, i, params)
            if outcome is None:
                unlabeled += 1
                continue
            counts[outcome.barrier] += 1
            positives += outcome.label

    labeled = sum(counts.values())
    horizontal = (counts["upper"] + counts["lower"]) / labeled if labeled else 0.0
    return LabelProfile(
        horizon=horizon,
        pt_mult=pt_mult,
        excess=excess,
        n_labeled=labeled,
        n_unlabeled=unlabeled,
        upper=round(counts["upper"] / labeled, 4) if labeled else 0.0,
        lower=round(counts["lower"] / labeled, 4) if labeled else 0.0,
        vertical=round(counts["vertical"] / labeled, 4) if labeled else 0.0,
        horizontal_share=round(horizontal, 4),
        positive_rate=round(positives / labeled, 4) if labeled else 0.0,
        in_target_band=TARGET_HORIZONTAL_LOW <= horizontal <= TARGET_HORIZONTAL_HIGH,
    )


def calibrate_barriers(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    horizon: int = LABEL_HORIZON,
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
    excess: bool = False,
) -> dict:
    """Scan barrier widths and recommend the one closest to a balanced mix."""
    profiles = [
        profile_labels(bars_by_symbol, horizon, mult, excess=excess) for mult in multipliers
    ]
    in_band = [p for p in profiles if p.in_target_band]
    target_mid = (TARGET_HORIZONTAL_LOW + TARGET_HORIZONTAL_HIGH) / 2
    best = (
        min(in_band, key=lambda p: abs(p.horizontal_share - target_mid))
        if in_band
        else min(profiles, key=lambda p: abs(p.horizontal_share - target_mid))
    )
    logger.info(
        "Barrier calibration",
        horizon=horizon,
        recommended=best.pt_mult,
        horizontal_share=best.horizontal_share,
    )
    return {
        "horizon": horizon,
        "excess": excess,
        "profiles": [p.as_dict() for p in profiles],
        "recommended_mult": best.pt_mult,
        "recommended_horizontal_share": best.horizontal_share,
        "target_band": [TARGET_HORIZONTAL_LOW, TARGET_HORIZONTAL_HIGH],
        "note": (
            "Label-shape measurement only — no model was fitted, so this costs "
            "nothing against the gate's n_trials."
        ),
    }


@dataclass(frozen=True)
class TargetScore:
    horizon: int
    excess: bool
    pt_mult: float
    horizontal_share: float
    positive_rate: float
    n_samples: int
    best_feature: str
    best_feature_ic: float
    mean_abs_ic: float

    def as_dict(self) -> dict:
        return asdict(self)


def score_targets(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
    excess_options: tuple[bool, ...] = (False, True),
    lookback: int = FEATURE_LOOKBACK,
    min_history: int = FULL_HISTORY,
    min_universe: int = 20,
    max_sessions: int = 500,
) -> dict:
    """Rank candidate targets by how well RAW FEATURES already predict them.

    For each candidate: calibrate the barrier width, build the labels, and
    measure the per-session IC of every raw feature rank against the label.
    A target no single feature ranks above chance is a target no model built on
    those features will rank either — and finding that out costs no model fits.
    """
    results: list[TargetScore] = []
    for excess in excess_options:
        for horizon in horizons:
            calibration = calibrate_barriers(
                bars_by_symbol, horizon=horizon, multipliers=multipliers, excess=excess
            )
            mult = float(calibration["recommended_mult"])
            score = _score_one_target(
                bars_by_symbol,
                LabelParams(pt_mult=mult, sl_mult=mult, horizon=horizon, excess=excess),
                float(calibration["recommended_horizontal_share"]),
                lookback=lookback,
                min_history=min_history,
                min_universe=min_universe,
                max_sessions=max_sessions,
            )
            if score is not None:
                results.append(score)

    ranked = sorted(results, key=lambda s: abs(s.best_feature_ic), reverse=True)
    return {
        "candidates": [s.as_dict() for s in ranked],
        "best": ranked[0].as_dict() if ranked else None,
        "note": (
            "Model-free: IC of raw feature ranks against each candidate label. "
            "Nothing was fitted, so no trials were consumed. A best_feature_ic "
            "indistinguishable from zero across every candidate means the "
            "features, not the target, are the problem."
        ),
    }


def _score_one_target(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    params: LabelParams,
    horizontal_share: float,
    lookback: int,
    min_history: int,
    min_universe: int,
    max_sessions: int,
) -> TargetScore | None:
    ordered_bars = {s: sorted(b, key=lambda x: x.timestamp) for s, b in bars_by_symbol.items()}
    prices = {s: adjusted_ohlc(b) for s, b in ordered_bars.items()}
    index_by_date = {
        s: {b.timestamp: i for i, b in enumerate(bars)} for s, bars in ordered_bars.items()
    }
    lengths = {s: len(p[3]) for s, p in prices.items()}
    market = _market_path({s: p[3] for s, p in prices.items()}, max(lengths.values()))

    all_dates = sorted({b.timestamp for bars in ordered_bars.values() for b in bars})[
        -max_sessions:
    ]

    per_session_ic: dict[str, list[float]] = {}
    n_samples = 0
    positives = 0
    for session in all_dates:
        snapshot, labels = [], []
        for symbol, bars in ordered_bars.items():
            i = index_by_date[symbol].get(session)
            if i is None or i + 1 < min_history:
                continue
            _, highs, lows, closes = prices[symbol]
            outcome = (
                excess_barrier_label(closes, market, i, params)
                if params.excess and len(market) == lengths[symbol]
                else triple_barrier_label(closes, highs, lows, i, params)
            )
            if outcome is None:
                continue
            snapshot.append(compute_feature_vector(bars[max(0, i - lookback + 1) : i + 1]))
            labels.append(float(outcome.label))
        if len(snapshot) < min_universe:
            continue
        ranked = cross_sectional_rank(snapshot)
        y = np.asarray(labels, dtype=float)
        n_samples += len(y)
        positives += int(y.sum())
        for name in ranked[0].features:
            column = np.array([r.features.get(name, 0.5) for r in ranked], dtype=float)
            per_session_ic.setdefault(name, []).append(spearman(column, y))

    if not per_session_ic or n_samples == 0:
        return None
    mean_ic = {name: float(np.mean(values)) for name, values in per_session_ic.items()}
    best_feature = max(mean_ic, key=lambda k: abs(mean_ic[k]))
    return TargetScore(
        horizon=params.horizon,
        excess=params.excess,
        pt_mult=params.pt_mult,
        horizontal_share=horizontal_share,
        positive_rate=round(positives / n_samples, 4),
        n_samples=n_samples,
        best_feature=best_feature,
        best_feature_ic=round(mean_ic[best_feature], 5),
        mean_abs_ic=round(float(np.mean([abs(v) for v in mean_ic.values()])), 5),
    )


__all__ = [
    "LabelProfile",
    "TargetScore",
    "calibrate_barriers",
    "profile_labels",
    "score_targets",
]
