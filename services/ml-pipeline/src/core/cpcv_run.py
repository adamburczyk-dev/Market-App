"""Run CPCV end to end and report the DISPERSION across paths (P4-4).

`run_training` answers "what did this model do out of sample". This answers
"how much would that number have changed had I cut the data differently", which
is the question run #2 made unavoidable: fold Sharpes from −1.61 to +4.54 on
identical data and an identical configuration.

The output leads with the spread, not the mean. A mean across paths with no
spread beside it is the single-path number wearing a different hat, and
`share_positive` — on how many of the ways of cutting the data the strategy
made money at all — is the number that decides whether there is anything here.

This is expensive: C(N, k) fits rather than one walk-forward pass. It is a
measurement run, not part of every training call, and like the config sweep it
raises `n_trials` — every path is another look at the same data.
"""

from datetime import datetime
from typing import Any

import numpy as np
import structlog

from src.core.cpcv import cpcv_splits, n_backtest_paths, path_dispersion
from src.core.dataset import Dataset
from src.core.evaluation import relative_metrics, top_quantile_portfolio
from src.core.training import TrainingParams, fit_on_dates

logger = structlog.get_logger()


def run_cpcv(
    ds: Dataset,
    params: TrainingParams | None = None,
    n_groups: int = 6,
    test_groups: int = 2,
) -> dict[str, Any]:
    """Fit every combination, assemble the paths, and report their dispersion."""
    p = params or TrainingParams()
    sessions = sorted(set(ds.dates))
    design = cpcv_splits(
        sessions,
        n_groups=n_groups,
        test_groups=test_groups,
        horizon=p.horizon,
        embargo=p.embargo,
    )
    if not design.paths:
        raise ValueError(
            f"{len(sessions)} sessions with {n_groups} groups produced no complete paths"
        )

    # predictions per split, kept per GROUP so a path can pick them up
    per_split: list[dict[int, tuple[list[datetime], np.ndarray, np.ndarray, list[str]]]] = []
    trained = 0
    for split in design.splits:
        model = fit_on_dates(ds, list(split.train_dates), p)
        if model is None:
            per_split.append({})
            continue
        trained += 1
        by_group: dict[int, tuple[list[datetime], np.ndarray, np.ndarray, list[str]]] = {}
        for group in split.groups:
            group_dates = _group_dates(sessions, design.n_groups, group)
            mask = np.array([d in group_dates for d in ds.dates], dtype=bool)
            if not mask.any():
                continue
            by_group[group] = (
                [d for d, m in zip(ds.dates, mask, strict=True) if m],
                model.predict_proba(ds.x[mask]),
                ds.next_returns[mask],
                [s for s, m in zip(ds.symbols, mask, strict=True) if m],
            )
        per_split.append(by_group)

    path_sharpe: list[float] = []
    path_active: list[float] = []
    path_ic: list[float] = []
    for path in design.paths:
        dates: list[datetime] = []
        symbols: list[str] = []
        probs: list[float] = []
        returns: list[float] = []
        complete = True
        for group, split_index in enumerate(path):
            piece = per_split[split_index].get(group)
            if piece is None:
                complete = False
                break
            d, pr, rt, sy = piece
            dates.extend(d)
            probs.extend(pr.tolist())
            returns.extend(rt.tolist())
            symbols.extend(sy)
        if not complete or not dates:
            continue
        order = np.argsort(np.asarray(dates))
        d_sorted = [dates[i] for i in order]
        s_sorted = [symbols[i] for i in order]
        pr_sorted = np.asarray(probs)[order]
        rt_sorted = np.asarray(returns)[order]

        tranches = p.horizon if p.overlapping_tranches else 1
        portfolio = top_quantile_portfolio(
            d_sorted, s_sorted, pr_sorted, rt_sorted, p.quantile, p.cost_bps, tranches
        )
        rel = relative_metrics(
            d_sorted, s_sorted, pr_sorted, rt_sorted, p.quantile, p.cost_bps, tranches
        )
        path_sharpe.append(portfolio.sharpe)
        path_active.append(rel.sharpe_active)
        path_ic.append(rel.ic_mean)

    active_stats = path_dispersion(path_active)
    report: dict[str, Any] = {
        "n_groups": design.n_groups,
        "test_groups": design.test_groups,
        "n_splits": len(design.splits),
        "n_splits_trained": trained,
        "n_paths_evaluated": len(path_sharpe),
        # P3-4: active leads, same as the gate.
        "sharpe_active": active_stats,
        "sharpe": path_dispersion(path_sharpe),
        "ic": path_dispersion(path_ic),
        "note": (
            "Dispersion across out-of-sample PATHS, not folds. Read share_positive "
            "before the mean: a strategy positive on 3 of 5 ways of cutting the same "
            "data has not been shown to work, whatever its average says. Every path "
            "is another look at the same data — raise the gate's n_trials."
        ),
    }
    logger.info(
        "CPCV finished",
        paths=len(path_sharpe),
        active_mean=active_stats["mean"],
        active_share_positive=active_stats.get("share_positive"),
    )
    return report


def _group_dates(sessions: list[datetime], n_groups: int, group: int) -> set[datetime]:
    bounds = np.linspace(0, len(sessions), n_groups + 1).astype(int)
    return set(sessions[int(bounds[group]) : int(bounds[group + 1])])


def cpcv_trials(n_groups: int, test_groups: int) -> int:
    """How many looks at the data a CPCV run costs, for `GateThresholds.n_trials`.

    Understating this makes the deflated Sharpe optimistic, which is the one
    place where being generous to ourselves is not a rounding error.
    """
    return max(1, n_backtest_paths(n_groups, test_groups))


__all__ = ["cpcv_trials", "run_cpcv"]
