"""Configuration sweep — does ANY training setup extract an out-of-sample edge?

The capacity probe answers whether the features contain structure at all. It
cannot say whether extracting that structure produces something tradable: a
model can fit its training window better and still rank the next cross-section
no better than chance. This module measures that directly — several training
configurations, each scored on the walk-forward folds.

Two rules keep the sweep honest:

- **the holdout never participates.** Selection happens on the fold windows
  only; the holdout stays untouched for the final gate, which is the only
  reason its numbers mean anything.
- **the number of configurations is carried out of the sweep** as `n_trials`,
  so the deflated Sharpe in the gate can be told the truth about how many times
  we looked. Trying twelve setups and then reporting the best as if it were the
  first is the standard way to manufacture an edge.

Ranking uses the IC t-statistic over the folds' cross-sections, not the fold
Sharpe: with 63 sessions per fold, Sharpe cannot distinguish the candidates.
"""

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import structlog

from src.core.dataset import Dataset
from src.core.gate import ic_tstat
from src.core.model import TrainConfig
from src.core.splits import purged_walk_forward
from src.core.training import (
    FoldReport,
    TrainingParams,
    fit_on_dates,
    score_window,
)

logger = structlog.get_logger()


# A deliberately small grid spanning the axes the run-#2 diagnosis implicates:
# capacity (the production net is tiny), regularization (dropout 0.3 on 7
# features is heavy), and how long the fit is allowed to run.
DEFAULT_GRID: dict[str, TrainConfig] = {
    "production": TrainConfig(),
    "wider": TrainConfig(hidden=(64, 32)),
    "wider_less_dropout": TrainConfig(hidden=(64, 32), dropout=0.1),
    "wide_low_reg": TrainConfig(hidden=(128, 64), dropout=0.1, weight_decay=1e-5),
    "longer_patient": TrainConfig(min_epochs=60, patience=30),
    "lower_lr": TrainConfig(lr=1e-3, min_epochs=60, patience=30),
}


@dataclass(frozen=True)
class CandidateResult:
    name: str
    n_folds: int
    auc_val_mean: float
    auc_train_mean: float
    ic_mean: float
    ic_tstat: float
    sharpe_mean: float
    lift_mean: float
    pred_std_mean: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SweepReport:
    candidates: list[CandidateResult] = field(default_factory=list)
    n_trials: int = 0  # feed this to the gate's deflated Sharpe
    best: str | None = None

    def as_dict(self) -> dict:
        return {
            "candidates": [c.as_dict() for c in self.candidates],
            "n_trials": self.n_trials,
            "best": self.best,
            "note": (
                "Selection used the walk-forward folds only; the holdout was not "
                "touched. Pass n_trials to the activation gate so G5 deflates for "
                "the number of configurations tried."
            ),
        }


def _aggregate(name: str, folds: list[FoldReport]) -> CandidateResult:
    """Pool a candidate's folds. IC is pooled across cross-sections, not averaged
    over folds, so a fold with fewer sessions cannot outvote a longer one."""
    ic_values: list[float] = []
    ic_stds: list[float] = []
    sections = 0
    for f in folds:
        if f.relative is None or f.relative.n_cross_sections < 2:
            continue
        ic_values.append(f.relative.ic_mean * f.relative.n_cross_sections)
        ic_stds.append(f.relative.ic_std)
        sections += f.relative.n_cross_sections
    ic_mean = sum(ic_values) / sections if sections else 0.0
    ic_std = float(np.mean(ic_stds)) if ic_stds else 0.0
    return CandidateResult(
        name=name,
        n_folds=len(folds),
        auc_val_mean=float(np.mean([f.auc for f in folds])) if folds else 0.5,
        auc_train_mean=float(np.mean([f.auc_train for f in folds])) if folds else 0.5,
        ic_mean=ic_mean,
        ic_tstat=ic_tstat(ic_mean, ic_std, sections),
        sharpe_mean=float(np.mean([f.portfolio.sharpe for f in folds])) if folds else 0.0,
        lift_mean=float(np.mean([f.diagnostics.lift for f in folds])) if folds else 0.0,
        pred_std_mean=float(np.mean([f.diagnostics.pred_std for f in folds])) if folds else 0.0,
    )


def run_sweep(
    ds: Dataset,
    params: TrainingParams | None = None,
    grid: dict[str, TrainConfig] | None = None,
    n_folds: int = 4,
) -> SweepReport:
    """Score each candidate config on the most recent ``n_folds`` walk-forward folds."""
    p = params or TrainingParams()
    configs = grid or DEFAULT_GRID
    sessions = sorted(set(ds.dates))
    if len(sessions) < p.holdout_size + p.train_size + p.test_size:
        raise ValueError(
            f"dataset has {len(sessions)} sessions; needs ≥ "
            f"{p.holdout_size + p.train_size + p.test_size} to sweep on folds alone"
        )

    work = sessions[: -p.holdout_size]  # the holdout is not part of selection
    folds = purged_walk_forward(work, p.train_size, p.test_size, p.horizon, p.embargo)[-n_folds:]
    if not folds:
        raise ValueError("no evaluable folds in the work window")

    results: list[CandidateResult] = []
    for name, config in configs.items():
        candidate_params = replace(p, model=config)
        reports: list[FoldReport] = []
        for k, fold in enumerate(folds):
            model = fit_on_dates(ds, list(fold.train_dates), candidate_params)
            if model is None:
                continue
            reports.append(
                score_window(
                    ds,
                    model,
                    set(fold.test_dates),
                    f"fold_{k}",
                    len(fold.train_dates),
                    candidate_params,
                    fit_dates=set(fold.train_dates),
                )
            )
        result = _aggregate(name, reports)
        results.append(result)
        logger.info(
            "Sweep candidate scored",
            candidate=name,
            auc_val=round(result.auc_val_mean, 4),
            auc_train=round(result.auc_train_mean, 4),
            ic_tstat=round(result.ic_tstat, 2),
        )

    ranked = sorted(results, key=lambda r: r.ic_tstat, reverse=True)
    best = ranked[0].name if ranked and ranked[0].ic_tstat > 0 else None
    return SweepReport(candidates=ranked, n_trials=len(configs), best=best)
