"""Meta-labeling: should we act on this signal at all? (P5-1, AFML ch. 3)

The base model answers "which names, ranked how". That is one question, and it
is being asked to carry a second one it was never trained on: whether a given
selection is worth trading after costs. Meta-labeling splits them:

    base model  ->  WHICH names, in which direction   (cross-sectional ranking)
    meta model  ->  ACT or SKIP on each such signal   (binary, on selected rows)

The meta-model trains only on rows the base model actually selected — that is
what makes it a filter rather than a second ranker — and its label is the one
thing the base model never sees: **did this trade make money net of costs**.
With P5-2 in place that cost is per name, so the filter can learn that a
marginal signal in an expensive name is not worth taking while the same signal
in a liquid one is.

**What this cannot do, and the reason it ships disabled.** Meta-labeling raises
PRECISION on a set of signals; it cannot create an edge that is not there. Run
on a base model with no ranking information, it fits noise, and its
out-of-sample "improvement" is the ordinary overfit of having looked at the data
again. The prediction plan makes this an explicit entry condition — gate
condition G1 (mean-IC t-statistic >= 2) — and until that is met the honest use
of this module is to measure, not to enable. It is also another look at the same
data, so `suggested_n_trials` comes back for the deflated Sharpe.

The number to read is NOT precision. A filter that lifts hit-rate from 52% to
70% while cutting the book from 400 trades to 12 has not improved anything it
can be paid for. The decision metric is the same one the gate uses — active
Sharpe against the equal-weight universe — reported next to how many trades
survive.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import numpy as np
import structlog

from src.core.evaluation import _annualized_sharpe, auc
from src.core.model import TrainConfig, train_classifier
from src.core.splits import purged_walk_forward

logger = structlog.get_logger()

BASE_PROBABILITY_FEATURE = "base_probability"


@dataclass(frozen=True)
class MetaParams:
    """How the filter is defined, trained and judged."""

    # Which rows count as "a signal": the same per-session top quantile the
    # book actually holds, so the filter sees exactly the trades it would veto.
    quantile: float = 0.2
    cost_bps: float = 5.0
    # Act when the meta-probability is at least this. 0.5 is the neutral
    # starting point; tuning it is another look at the data and must be paid
    # for in n_trials like any other.
    threshold: float = 0.5
    horizon: int = 10
    embargo: int = 5
    # Same shape as the base walk-forward, so the filter is judged over the same
    # kind of window the model it filters was judged over.
    train_size: int = 504
    test_size: int = 63
    min_signals: int = 200
    config: TrainConfig = TrainConfig()


@dataclass(frozen=True)
class MetaDataset:
    x: np.ndarray
    y: np.ndarray  # 1 = the trade made money net of costs
    dates: list[datetime]
    symbols: list[str]
    net_returns: np.ndarray
    feature_names: list[str]

    @property
    def n_signals(self) -> int:
        return int(self.x.shape[0])


def build_meta_dataset(
    x: np.ndarray,
    feature_names: list[str],
    dates: list[datetime],
    symbols: list[str],
    base_probabilities: np.ndarray,
    forward_returns: np.ndarray,
    params: MetaParams | None = None,
    cost_bps_by_symbol: dict[str, float] | None = None,
) -> MetaDataset:
    """Rows the base model SELECTED, labelled by their net-of-cost outcome.

    Selecting per session rather than globally matters: the book holds a
    quantile of each cross-section, so a global threshold would train the filter
    on a different set of trades than the one it is meant to police.

    The base model's own probability is appended as a feature. Without it the
    filter cannot express "act on the strong signals, skip the marginal ones",
    which is the most obvious thing it might learn.
    """
    p = params or MetaParams()
    by_date: dict[datetime, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        by_date[d].append(i)

    selected: list[int] = []
    for rows in by_date.values():
        k = max(1, math.ceil(p.quantile * len(rows)))
        selected.extend(sorted(rows, key=lambda i: float(base_probabilities[i]), reverse=True)[:k])
    selected.sort()

    if not selected:
        return MetaDataset(
            np.zeros((0, len(feature_names) + 1)),
            np.zeros(0),
            [],
            [],
            np.zeros(0),
            [*feature_names, BASE_PROBABILITY_FEATURE],
        )

    rows_x = np.column_stack([x[selected], np.asarray(base_probabilities, dtype=float)[selected]])
    costs = np.array(
        [(cost_bps_by_symbol or {}).get(symbols[i], p.cost_bps) / 10_000.0 for i in selected],
        dtype=float,
    )
    net = np.asarray(forward_returns, dtype=float)[selected] - costs
    return MetaDataset(
        x=rows_x,
        y=(net > 0).astype(float),
        dates=[dates[i] for i in selected],
        symbols=[symbols[i] for i in selected],
        net_returns=net,
        feature_names=[*feature_names, BASE_PROBABILITY_FEATURE],
    )


def _book_returns(
    dates: list[datetime], net_returns: np.ndarray, keep: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-session mean net return of the whole signal set and of the kept subset.

    A session where the filter vetoes everything contributes a flat 0.0 rather
    than being dropped: refusing to trade is a real outcome with a real return,
    and skipping those sessions would quietly compare the filtered book only on
    the days it chose to show up.
    """
    by_date: dict[datetime, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        by_date[d].append(i)
    sessions = sorted(by_date)
    unfiltered = np.array([float(np.mean(net_returns[by_date[s]])) for s in sessions], dtype=float)
    filtered = np.array(
        [
            float(np.mean(net_returns[[i for i in by_date[s] if keep[i]]]))
            if any(keep[i] for i in by_date[s])
            else 0.0
            for s in sessions
        ],
        dtype=float,
    )
    return unfiltered, filtered


def run_meta_labeling(
    meta: MetaDataset,
    params: MetaParams | None = None,
) -> dict[str, Any]:
    """Fit the filter out of sample and report whether it earns its place.

    Purged walk-forward over the signal dates — the labels span `horizon`
    sessions exactly like the base model's, so the same leakage applies and the
    same purge is required.
    """
    p = params or MetaParams()
    if meta.n_signals < p.min_signals:
        raise ValueError(
            f"{meta.n_signals} signals is too few to fit a filter on (need >= {p.min_signals})"
        )
    if len(set(meta.y.tolist())) < 2:
        raise ValueError("every signal has the same outcome — nothing to learn")

    sessions = sorted(set(meta.dates))
    folds = purged_walk_forward(sessions, p.train_size, p.test_size, p.horizon, p.embargo)
    if not folds:
        raise ValueError(
            f"{len(sessions)} signal sessions cannot support a "
            f"{p.train_size}+{p.test_size} walk-forward"
        )
    probabilities = np.full(meta.n_signals, np.nan, dtype=float)
    for fold in folds:
        train_dates, test_dates = set(fold.train_dates), set(fold.test_dates)
        train_mask = np.array([d in train_dates for d in meta.dates], dtype=bool)
        test_mask = np.array([d in test_dates for d in meta.dates], dtype=bool)
        if train_mask.sum() < 50 or not test_mask.any():
            continue
        if len(set(meta.y[train_mask].tolist())) < 2:
            continue
        split = max(1, int(train_mask.sum() * 0.8))
        indices = np.flatnonzero(train_mask)
        fit_rows, val_rows = indices[:split], indices[split:]
        if len(val_rows) < 10 or len(set(meta.y[val_rows].tolist())) < 2:
            continue
        model = train_classifier(
            meta.x[fit_rows],
            meta.y[fit_rows],
            meta.x[val_rows],
            meta.y[val_rows],
            meta.feature_names,
            replace(p.config, seed=p.config.seed),
        )
        probabilities[test_mask] = model.predict_proba(meta.x[test_mask])

    scored = np.isfinite(probabilities)
    if not scored.any():
        raise ValueError("no fold produced out-of-sample meta-predictions")

    dates = [d for d, s in zip(meta.dates, scored, strict=True) if s]
    net = meta.net_returns[scored]
    y = meta.y[scored]
    prob = probabilities[scored]
    keep = prob >= p.threshold

    unfiltered, filtered = _book_returns(dates, net, keep)
    kept = int(keep.sum())
    report: dict[str, Any] = {
        "n_signals": meta.n_signals,
        "n_scored": int(scored.sum()),
        "n_kept": kept,
        "kept_share": round(kept / int(scored.sum()), 4),
        "threshold": p.threshold,
        "base_hit_rate": round(float(y.mean()), 4),
        "filtered_hit_rate": round(float(y[keep].mean()), 4) if kept else None,
        "precision_lift": round(float(y[keep].mean() - y.mean()), 4) if kept else None,
        "meta_auc": round(auc(y, prob), 4),
        # the decision numbers — precision is not one of them
        "sharpe_unfiltered": round(_annualized_sharpe(unfiltered), 3),
        "sharpe_filtered": round(_annualized_sharpe(filtered), 3),
        "sharpe_delta": round(_annualized_sharpe(filtered) - _annualized_sharpe(unfiltered), 3),
        "suggested_n_trials": len(folds),
        "verdict": "",
    }
    report["verdict"] = _verdict(report)
    logger.info(
        "Meta-labeling finished",
        signals=meta.n_signals,
        kept_share=report["kept_share"],
        sharpe_delta=report["sharpe_delta"],
    )
    return report


def _verdict(report: dict[str, Any]) -> str:
    """Whether the filter earned its place — judged on the book, not on precision."""
    delta = float(report["sharpe_delta"])
    kept_share = float(report["kept_share"])
    lift = report["precision_lift"]
    meta_auc = float(report["meta_auc"])

    if meta_auc <= 0.5:
        return (
            f"The filter does not discriminate at all (meta AUC {meta_auc:.3f}). Any "
            "change in the book is the threshold cutting trades at random, not a "
            "decision. Do not enable."
        )
    if kept_share < 0.1:
        return (
            f"The filter keeps only {kept_share:.0%} of signals. Whatever it did to "
            "the Sharpe, a book this thin is a different strategy rather than a "
            "filtered one, and its statistics carry almost no evidence."
        )
    if delta <= 0:
        return (
            f"Precision lift {lift:+.3f} but the book got no better "
            f"(Sharpe {report['sharpe_unfiltered']:.2f} -> "
            f"{report['sharpe_filtered']:.2f}). Skipping trades that were going to "
            "lose is only worth it if the ones kept pay for the ones missed. "
            "Do not enable."
        )
    return (
        f"The filter keeps {kept_share:.0%} of signals, lifts the hit rate by "
        f"{lift:+.3f} and the Sharpe from {report['sharpe_unfiltered']:.2f} to "
        f"{report['sharpe_filtered']:.2f}. Worth enabling ONLY if the base model "
        "already clears gate condition G1 — on a base model without ranking "
        "information this same number is what overfitting looks like."
    )


__all__ = [
    "BASE_PROBABILITY_FEATURE",
    "MetaDataset",
    "MetaParams",
    "build_meta_dataset",
    "run_meta_labeling",
]
