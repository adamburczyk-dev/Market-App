"""How fast the signal decays, and how fast we must act on it (P5-4).

Two decisions have so far been made by assumption rather than by measurement:

**How long to hold.** The overlapping-tranche construction (T0-4) holds a
position for `horizon` sessions because that is the label's horizon. That is a
coherent default, not evidence. If the predictive content of the features peaks
at 21 sessions, a 10-session hold sells the position halfway through the move;
if it is gone by 5, a 10-session label is mostly measuring noise and the tranche
count is buying turnover for nothing.

**How urgently to enter.** Live, a signal is acted on some time after the data
that produced it. If a one-session delay costs most of the IC, the whole design
has an execution-latency problem that no amount of model work will fix; if it
costs nothing, entering patiently is free and the cost model (P5-2) says to do
exactly that.

Both are answered with the same instrument and no model: the per-session rank
IC of each RAW feature against forward returns measured over h sessions,
starting d sessions later. Model-free means no fit, no `n_trials` consumed, and
no possibility that the answer is really a statement about one particular model.

The t-statistic is what to read, not the IC level — over 63 sessions with
overlapping windows, an IC of 0.02 and one of −0.02 are the same observation
seen twice.

**How much to trust the t-statistic here, and the null this study carries with
it.** The t-statistic gets a Newey-West correction for the fact that
consecutive sessions' forward windows overlap by h-1 sessions — without it,
five features on a universe of independent random walks looked significant at
|t| ~ 5. That correction is exact for the overlap it models and INCOMPLETE for
the other source of persistence: the features are themselves slow-moving
(`price_to_sma50` spans fifty sessions), so their IC series stays correlated
well past the h-1 lag it covers. Measured on random walks, 4 of 5 null
universes still cleared a fixed |t| >= 2 bar.

A fixed threshold therefore cannot be the test, so the study computes its own,
the way the capacity probe uses shuffled labels: the feature panel's SYMBOL
labels are permuted, which preserves each name's feature persistence and the
return overlap exactly while destroying the pairing that carries any real
signal. The largest |t| that survives that permutation is the bar the real
numbers have to clear. It is reported as `null_tstat` — read it as "this is
what noise scores on this data", and note that it rises with the horizon,
which is precisely the bias a fixed bar would have missed.
"""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import structlog
from trading_common.prices import adjusted_closes
from trading_common.schemas import OHLCVBar

from src.core.dataset import Dataset, DatasetParams, build_dataset
from src.core.evaluation import FeatureIC, per_feature_ic

logger = structlog.get_logger()

DEFAULT_HORIZONS = (1, 5, 10, 21, 63)
DEFAULT_DELAYS = (0, 1, 2, 3, 5)
# Below this the IC series is too short for its t-statistic to mean anything.
MIN_CROSS_SECTIONS = 30


@dataclass(frozen=True)
class DecayPoint:
    horizon: int
    delay: int
    ic_mean: float
    ic_tstat: float
    n_cross_sections: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "delay": self.delay,
            "ic": round(self.ic_mean, 5),
            "t": round(self.ic_tstat, 2),
            "n": self.n_cross_sections,
        }


def _forward_returns(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    dates: list,
    symbols: list[str],
    horizon: int,
    delay: int,
) -> np.ndarray:
    """Return from close[t+delay] to close[t+delay+horizon], NaN where unavailable.

    Adjusted closes, because this is measuring returns — the same series the
    features and labels are built on (P0-1).
    """
    index_by_symbol: dict[str, dict[Any, int]] = {}
    closes_by_symbol: dict[str, np.ndarray] = {}
    for symbol, bars in bars_by_symbol.items():
        ordered = sorted(bars, key=lambda b: b.timestamp)
        index_by_symbol[symbol] = {b.timestamp.date(): i for i, b in enumerate(ordered)}
        closes_by_symbol[symbol] = adjusted_closes(ordered)

    out = np.full(len(dates), np.nan, dtype=float)
    for row, (date, symbol) in enumerate(zip(dates, symbols, strict=True)):
        closes = closes_by_symbol.get(symbol)
        if closes is None:
            continue
        start = index_by_symbol[symbol].get(date.date())
        if start is None:
            continue
        entry, exit_ = start + delay, start + delay + horizon
        if exit_ >= len(closes) or closes[entry] <= 0:
            continue
        out[row] = float(closes[exit_] / closes[entry] - 1.0)
    return out


def _ic_table(
    ds: Dataset,
    bars_by_symbol: dict[str, list[OHLCVBar]],
    horizon: int,
    delay: int,
) -> dict[str, FeatureIC]:
    """Standalone IC of every raw feature against the (horizon, delay) return.

    Reuses the public E2 instrument rather than reimplementing it, so a decay
    curve and the per-feature IC table in the training report can never come to
    disagree about what "the IC of this feature" means. Rows whose forward
    window runs past the end of the history are dropped, not zero-filled — a
    truncated window is missing data, and calling it a zero return would drag
    every long-horizon IC toward nothing.

    ``overlap=horizon`` is load-bearing. Consecutive sessions' ICs share h-1
    sessions of their forward window, so the naive standard error is too small
    by roughly sqrt(h) — and worse for this study specifically, too small BY
    MORE at longer horizons, which would tilt the holding-period recommendation
    toward the longest horizon tested for a purely mechanical reason. Without
    it, a universe of independent random walks reported six features at |t| >= 2
    and a confident 10-session recommendation.
    """
    forward = _forward_returns(bars_by_symbol, ds.dates, ds.symbols, horizon, delay)
    usable = np.isfinite(forward)
    if not usable.any():
        return {}
    dates = [d for d, keep in zip(ds.dates, usable, strict=True) if keep]
    return per_feature_ic(dates, ds.x[usable], ds.feature_names, forward[usable], overlap=horizon)


def _permuted_symbols(symbols: list[str], rng: np.random.Generator) -> dict[str, str]:
    """A derangement of the symbol labels — no name may keep its own features."""
    names = sorted(set(symbols))
    if len(names) < 2:
        return {name: name for name in names}
    for _ in range(50):
        shuffled = list(names)
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(names, shuffled, strict=True)):
            return dict(zip(names, shuffled, strict=True))
    return dict(zip(names, [*names[1:], names[0]], strict=True))  # cyclic fallback


def null_tstat(
    ds: Dataset,
    bars_by_symbol: dict[str, list[OHLCVBar]],
    horizons: tuple[int, ...],
    n_permutations: int = 3,
    seed: int = 17,
) -> float:
    """Largest |t| that noise produces on THIS data — the bar the real numbers clear.

    Permuting the symbol labels of the feature panel keeps every name's feature
    persistence and every forward window's overlap intact, and removes only the
    pairing between a name's features and its own returns. Whatever |t| survives
    that is what the study's own machinery scores on data with no signal, which
    is the only honest threshold: a fixed |t| >= 2 was cleared by 4 of 5 null
    universes.
    """
    rng = np.random.default_rng(seed)
    row_of = {(d, s): i for i, (d, s) in enumerate(zip(ds.dates, ds.symbols, strict=True))}
    worst = 0.0
    for _ in range(max(1, n_permutations)):
        mapping = _permuted_symbols(ds.symbols, rng)
        source = [row_of.get((d, mapping[s])) for d, s in zip(ds.dates, ds.symbols, strict=True)]
        keep = np.array([i is not None for i in source], dtype=bool)
        if keep.sum() < MIN_CROSS_SECTIONS:
            continue
        rows = np.array([i for i in source if i is not None], dtype=int)
        shuffled = replace(
            ds,
            x=ds.x[rows],
            dates=[d for d, k in zip(ds.dates, keep, strict=True) if k],
            symbols=[s for s, k in zip(ds.symbols, keep, strict=True) if k],
        )
        # the returns must stay with the ORIGINAL name, so score against bars
        for horizon in horizons:
            table = _ic_table(shuffled, bars_by_symbol, horizon, 0)
            for ic in table.values():
                worst = max(worst, abs(ic.tstat))
    return worst


def _profile(
    tables: dict[tuple[int, int], dict[str, FeatureIC]],
    feature: str,
    horizons: tuple[int, ...],
    delays: tuple[int, ...],
) -> tuple[list[DecayPoint], list[DecayPoint]]:
    def point(horizon: int, delay: int) -> DecayPoint:
        ic = tables.get((horizon, delay), {}).get(feature)
        if ic is None:
            return DecayPoint(horizon, delay, 0.0, 0.0, 0)
        return DecayPoint(horizon, delay, ic.mean, ic.tstat, ic.n_sessions)

    holding = [point(h, 0) for h in horizons]
    # urgency is measured at the horizon where the signal is strongest, since
    # that is the horizon anyone would actually trade
    best = max(holding, key=lambda p: abs(p.ic_tstat))
    return holding, [point(best.horizon, d) for d in delays]


def _half_life(points: list[DecayPoint]) -> float | None:
    """Horizon at which |IC| has fallen to half its peak, linearly interpolated.

    ``None`` when it never does within the tested range — which is itself an
    answer: the signal has not decayed by the longest horizon measured, so the
    holding period is not what is limiting it.
    """
    if not points:
        return None
    peak = max(points, key=lambda p: abs(p.ic_mean))
    if abs(peak.ic_mean) <= 0:
        return None
    target = abs(peak.ic_mean) / 2.0
    after = [p for p in points if p.horizon > peak.horizon]
    previous = peak
    for point in sorted(after, key=lambda p: p.horizon):
        if abs(point.ic_mean) <= target:
            span = point.horizon - previous.horizon
            drop = abs(previous.ic_mean) - abs(point.ic_mean)
            if drop <= 0:
                return float(point.horizon)
            return float(previous.horizon + span * (abs(previous.ic_mean) - target) / drop)
        previous = point
    return None


def run_alpha_decay(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    params: DatasetParams | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    delays: tuple[int, ...] = DEFAULT_DELAYS,
    max_features: int = 6,
    n_permutations: int = 3,
) -> dict[str, Any]:
    """Decay profile of the strongest raw features — holding period and urgency.

    The features are ranked by their standalone evidence at the current label
    horizon and only the strongest handful are profiled: a decay curve for a
    feature that predicts nothing at any horizon is a plot of noise, and printing
    fifteen of them buries the two that matter.
    """
    if not bars_by_symbol:
        raise ValueError("no history for any requested symbol")
    p = params or DatasetParams()
    ds = build_dataset(bars_by_symbol, p)
    if ds.n_samples == 0:
        raise ValueError("dataset is empty — not enough history for the feature window")

    base_horizon = p.label.horizon
    tables: dict[tuple[int, int], dict[str, FeatureIC]] = {
        (h, 0): _ic_table(ds, bars_by_symbol, h, 0) for h in horizons
    }
    if base_horizon not in horizons:
        tables[(base_horizon, 0)] = _ic_table(ds, bars_by_symbol, base_horizon, 0)

    # Pick which features are worth profiling at the CURRENT horizon — the one
    # the model is trained on. Choosing them by their best horizon instead would
    # be selecting on the same quantity the study then reports.
    ranked = [
        (name, abs(ic.tstat))
        for name, ic in tables[(base_horizon, 0)].items()
        if ic.n_sessions >= MIN_CROSS_SECTIONS
        and float(np.std(ds.x[:, ds.feature_names.index(name)])) > 0
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    chosen = [name for name, _ in ranked[:max_features]]

    for name in chosen:
        best_holding = max(
            (tables[(h, 0)].get(name) for h in horizons),
            key=lambda ic: abs(ic.tstat) if ic else 0.0,
            default=None,
        )
        if best_holding is None:
            continue
        best_h = next(h for h in horizons if tables[(h, 0)].get(name) is best_holding)
        for d in delays:
            if (best_h, d) not in tables:
                tables[(best_h, d)] = _ic_table(ds, bars_by_symbol, best_h, d)

    features: list[dict[str, Any]] = []
    for name in chosen:
        holding, urgency = _profile(tables, name, horizons, delays)
        peak = max(holding, key=lambda point: abs(point.ic_mean))
        immediate = next((u for u in urgency if u.delay == 0), None)
        retained = (
            [
                {
                    "delay": u.delay,
                    "ic_retained": round(u.ic_mean / immediate.ic_mean, 3)
                    if immediate and immediate.ic_mean != 0
                    else 0.0,
                }
                for u in urgency
            ]
            if immediate
            else []
        )
        features.append(
            {
                "feature": name,
                "holding": [point.as_dict() for point in holding],
                "peak_horizon": peak.horizon,
                "peak_ic": round(peak.ic_mean, 5),
                "peak_t": round(peak.ic_tstat, 2),
                "half_life_sessions": _half_life(holding),
                "urgency": [point.as_dict() for point in urgency],
                "ic_retained_by_delay": retained,
            }
        )

    noise_bar = null_tstat(ds, bars_by_symbol, horizons, n_permutations=n_permutations)
    report: dict[str, Any] = {
        "symbols": len(bars_by_symbol),
        "sessions": len(set(ds.dates)),
        "samples": ds.n_samples,
        "current_horizon": base_horizon,
        "horizons_tested": list(horizons),
        "delays_tested": list(delays),
        "features_profiled": chosen,
        "features": features,
        # what this study's own machinery scores on data with no signal
        "null_tstat": round(noise_bar, 2),
        "n_permutations": n_permutations,
        "verdict": _verdict(features, base_horizon, noise_bar),
    }
    logger.info(
        "Alpha decay finished",
        features=len(features),
        current_horizon=base_horizon,
        strongest=chosen[0] if chosen else None,
    )
    return report


def _verdict(features: list[dict[str, Any]], current_horizon: int, noise_bar: float = 2.0) -> str:
    """What the profile implies for the holding period and for execution.

    The bar is the LARGER of |t| = 2 and what the symbol-permuted null scored on
    this same data. A fixed 2 was cleared by 4 of 5 universes of independent
    random walks, so on its own it certifies noise.
    """
    bar = max(2.0, float(noise_bar))
    credible = [f for f in features if abs(float(f["peak_t"])) > bar]
    if not credible:
        return (
            f"No feature beats the noise bar (|t| > {bar:.1f}, where {noise_bar:.1f} is "
            "what this study scores on the same data with the symbol labels permuted), "
            "so there is no decay profile to read — the holding period is not what is "
            f"limiting this, and keeping the current {current_horizon}-session horizon "
            "costs nothing. Re-run once a feature family clears the E2 bar."
        )
    peaks = [int(f["peak_horizon"]) for f in credible]
    typical = int(np.median(peaks))
    parts = [
        f"{len(credible)} feature(s) carry credible evidence; their IC peaks at "
        f"{typical} sessions (range {min(peaks)}-{max(peaks)})."
    ]
    if typical > current_horizon * 1.5:
        parts.append(
            f"That is materially longer than the current {current_horizon}-session "
            "horizon — the label is being closed while the move is still running, "
            "and a longer horizon would both harvest more and cost less turnover."
        )
    elif typical * 1.5 < current_horizon:
        parts.append(
            f"That is materially shorter than the current {current_horizon}-session "
            "horizon — the label is measuring mostly what happens after the signal "
            "has already decayed."
        )
    else:
        parts.append(f"The current {current_horizon}-session horizon is consistent with that.")

    worst_retention = min(
        (
            float(entry["ic_retained"])
            for f in credible
            for entry in f["ic_retained_by_delay"]
            if int(entry["delay"]) == 1
        ),
        default=1.0,
    )
    if worst_retention < 0.7:
        parts.append(
            f"Entry is urgent: one session of delay retains only {worst_retention:.0%} "
            "of the IC, so execution latency is a first-order design constraint."
        )
    else:
        parts.append(
            f"Entry is not urgent: a one-session delay retains {worst_retention:.0%} "
            "of the IC, so patient execution is close to free and the cost model "
            "should decide the schedule."
        )
    return " ".join(parts)


__all__ = ["DEFAULT_DELAYS", "DEFAULT_HORIZONS", "DecayPoint", "run_alpha_decay"]
