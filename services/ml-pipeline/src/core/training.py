"""Purged walk-forward training + the activation-gate report (plan §6–§7).

The holdout — the most recent ``holdout_size`` sessions — is never touched
during model selection. Walk-forward folds run over the remaining history;
each fold trains on its purged window (with an internal, also-purged
train/val split for early stopping + calibration) and is scored on its test
block. The gate: cost-adjusted OOS Sharpe > threshold on the holdout AND on
at least 2 of the 3 most recent folds, with sane calibration (Brier no worse
than the base rate). Only a gate-passing model may serve non-HOLD signals.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import structlog

from src.core.dataset import Dataset
from src.core.evaluation import (
    PortfolioResult,
    RelativeMetrics,
    SelectionDiagnostics,
    auc,
    baseline_feature_ic,
    brier,
    relative_metrics,
    selection_diagnostics,
    top_quantile_portfolio,
)
from src.core.model import TrainConfig, TrainedModel, train_classifier
from src.core.splits import purged_walk_forward

logger = structlog.get_logger()


@dataclass(frozen=True)
class TrainingParams:
    train_size: int = 756  # ~3y of sessions
    test_size: int = 63  # ~3m
    holdout_size: int = 126  # ~6m, untouched during selection
    val_size: int = 63  # tail of each train window used for early stop + calibration
    horizon: int = 10  # label horizon → purge width
    embargo: int = 5
    quantile: float = 0.2
    cost_bps: float = 5.0
    gate_sharpe: float = 0.5
    baseline_feature: str = "return_20d"  # single-feature yardstick (T0-5)
    model: TrainConfig = field(default_factory=TrainConfig)


@dataclass(frozen=True)
class FoldReport:
    name: str
    n_train: int
    n_test: int
    auc: float
    brier: float
    portfolio: PortfolioResult
    diagnostics: SelectionDiagnostics
    # T0-3: AUC on the window the model was FITTED on, plus what the fit itself
    # did. auc_train ~ 0.5 means the model could not learn; auc_train high with
    # auc ~ 0.5 means it memorised. The out-of-sample numbers look identical in
    # both cases, and the responses are opposite.
    auc_train: float = 0.5
    fit: dict[str, Any] = field(default_factory=dict)
    # T0-5: benchmark-relative and rank-based measurement. A long-only Sharpe
    # in a window where most names rose says more about the market than the
    # model; IC/ICIR and the long-short leg do not have that failure mode.
    relative: RelativeMetrics | None = None
    baseline_ic: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GateReport:
    folds: list[FoldReport]
    holdout: FoldReport
    passed: bool
    reasons: list[str]

    def as_dict(self) -> dict:
        def fold(f: FoldReport) -> dict:
            return {
                "name": f.name,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "auc": round(f.auc, 4),
                "brier": round(f.brier, 4),
                "sharpe": round(f.portfolio.sharpe, 4),
                "mean_daily_return": round(f.portfolio.mean_daily_return, 6),
                "avg_turnover": round(f.portfolio.avg_turnover, 4),
                "avg_positions": round(f.portfolio.avg_positions, 2),
                "n_portfolio_sessions": f.portfolio.n_sessions,
                # discrimination diagnostics — separate real edge from luck
                "base_rate": round(f.diagnostics.base_rate, 4),
                "selected_hit_rate": round(f.diagnostics.selected_hit_rate, 4),
                "lift": round(f.diagnostics.lift, 4),
                "pred_mean": round(f.diagnostics.pred_mean, 4),
                "pred_std": round(f.diagnostics.pred_std, 4),
                "pred_p10": round(f.diagnostics.pred_p10, 4),
                "pred_p90": round(f.diagnostics.pred_p90, 4),
                # underfit-vs-no-signal diagnostics (T0-3)
                "auc_train": round(f.auc_train, 4),
                **f.fit,
                **(
                    {
                        "ic_mean": round(f.relative.ic_mean, 5),
                        "ic_std": round(f.relative.ic_std, 5),
                        "icir": round(f.relative.icir, 4),
                        "ic_positive_share": round(f.relative.ic_positive_share, 4),
                        "n_cross_sections": f.relative.n_cross_sections,
                        "sharpe_benchmark_ew": round(f.relative.sharpe_benchmark_ew, 4),
                        "sharpe_active": round(f.relative.sharpe_active, 4),
                        "sharpe_long_short": round(f.relative.sharpe_long_short, 4),
                        "sharpe_gross": round(f.relative.sharpe_gross, 4),
                        "sharpe_net": round(f.relative.sharpe_net, 4),
                        "cost_drag_annualized": round(f.relative.cost_drag_annualized, 5),
                        "turnover_daily_mean": round(f.relative.turnover_daily_mean, 4),
                    }
                    if f.relative is not None
                    else {}
                ),
                **{f"baseline_ic_{k}": round(v, 5) for k, v in f.baseline_ic.items()},
            }

        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "holdout": fold(self.holdout),
            "folds": [fold(f) for f in self.folds],
        }


def _mask(dates: list[datetime], allowed: set[datetime]) -> np.ndarray:
    return np.array([d in allowed for d in dates], dtype=bool)


def _fit_on_dates(
    ds: Dataset, dates: list[datetime], params: TrainingParams
) -> TrainedModel | None:
    """Train with the window's tail as the (purged) validation fold."""
    gap = params.horizon + params.embargo
    if len(dates) < params.val_size + gap + 20:  # need a real fit set left over
        return None
    fit_dates = set(dates[: -(params.val_size + gap)])
    val_dates = set(dates[-params.val_size :])
    fit_mask = _mask(ds.dates, fit_dates)
    val_mask = _mask(ds.dates, val_dates)
    if fit_mask.sum() == 0 or val_mask.sum() == 0:
        return None
    if len(np.unique(ds.y[fit_mask])) < 2:  # single-class window is untrainable
        return None
    return train_classifier(
        ds.x[fit_mask],
        ds.y[fit_mask],
        ds.x[val_mask],
        ds.y[val_mask],
        ds.feature_names,
        params.model,
    )


def _score(
    ds: Dataset,
    model: TrainedModel,
    test_dates: set[datetime],
    name: str,
    n_train: int,
    params: TrainingParams,
    fit_dates: set[datetime] | None = None,
) -> FoldReport:
    mask = _mask(ds.dates, test_dates)
    probs = model.predict_proba(ds.x[mask])
    dates = [d for d, m in zip(ds.dates, mask, strict=True) if m]
    portfolio = top_quantile_portfolio(
        dates,
        [s for s, m in zip(ds.symbols, mask, strict=True) if m],
        probs,
        ds.next_returns[mask],
        quantile=params.quantile,
        cost_bps=params.cost_bps,
    )
    relative = relative_metrics(
        dates,
        [s for s, m in zip(ds.symbols, mask, strict=True) if m],
        probs,
        ds.next_returns[mask],
        quantile=params.quantile,
        cost_bps=params.cost_bps,
    )
    # The rule: a model that cannot beat the raw rank of one feature has not
    # earned the ML layer. return_20d is the strongest single candidate here.
    baseline_ic: dict[str, float] = {}
    if params.baseline_feature in ds.feature_names:
        column = ds.x[mask][:, ds.feature_names.index(params.baseline_feature)]
        baseline_ic[params.baseline_feature] = baseline_feature_ic(
            dates, column, ds.next_returns[mask]
        )

    auc_train = 0.5
    if fit_dates:
        fit_mask = _mask(ds.dates, fit_dates)
        if fit_mask.sum() > 0:
            auc_train = auc(ds.y[fit_mask], model.predict_proba(ds.x[fit_mask]))
    return FoldReport(
        name=name,
        n_train=n_train,
        n_test=int(mask.sum()),
        auc=auc(ds.y[mask], probs),
        brier=brier(ds.y[mask], probs),
        portfolio=portfolio,
        diagnostics=selection_diagnostics(dates, ds.y[mask], probs, quantile=params.quantile),
        auc_train=auc_train,
        fit=dict(model.diagnostics),
        relative=relative,
        baseline_ic=baseline_ic,
    )


def run_training(
    ds: Dataset, params: TrainingParams | None = None
) -> tuple[TrainedModel, GateReport]:
    """Walk-forward evaluation → gate report → final model.

    The returned model is trained on ALL available history (with the standard
    internal val split) regardless of the gate outcome — the caller decides
    what a failed gate means (register as non-production, keep serving HOLD).
    Raises ValueError when the dataset is too small to evaluate at all.
    """
    p = params or TrainingParams()
    sessions = sorted(set(ds.dates))
    if len(sessions) < p.holdout_size + p.train_size + p.test_size:
        raise ValueError(
            f"dataset has {len(sessions)} sessions; needs ≥ "
            f"{p.holdout_size + p.train_size + p.test_size} for holdout + one fold"
        )

    work = sessions[: -p.holdout_size]
    holdout = sessions[-p.holdout_size :]

    folds = purged_walk_forward(work, p.train_size, p.test_size, p.horizon, p.embargo)
    fold_reports: list[FoldReport] = []
    for k, fold in enumerate(folds):
        model = _fit_on_dates(ds, list(fold.train_dates), p)
        if model is None:
            logger.warning("Fold skipped — untrainable window", fold=k)
            continue
        fold_reports.append(
            _score(
                ds,
                model,
                set(fold.test_dates),
                f"fold_{k}",
                len(fold.train_dates),
                p,
                fit_dates=set(fold.train_dates),
            )
        )

    # Holdout model: trained on everything BEFORE the holdout, purged at the seam.
    gap = p.horizon + p.embargo
    holdout_train = work[:-gap] if gap else work
    holdout_model = _fit_on_dates(ds, holdout_train, p)
    if holdout_model is None:
        raise ValueError("holdout window is untrainable (too small or single-class)")
    holdout_report = _score(
        ds,
        holdout_model,
        set(holdout),
        "holdout",
        len(holdout_train),
        p,
        fit_dates=set(holdout_train),
    )

    reasons: list[str] = []
    if holdout_report.portfolio.sharpe <= p.gate_sharpe:
        reasons.append(
            f"holdout sharpe {holdout_report.portfolio.sharpe:.2f} ≤ gate {p.gate_sharpe}"
        )
    recent = fold_reports[-3:]
    passing = sum(1 for f in recent if f.portfolio.sharpe > p.gate_sharpe)
    if len(recent) < 2:
        reasons.append(f"only {len(recent)} evaluable folds — need ≥ 2")
    elif passing < 2:
        reasons.append(f"only {passing}/{len(recent)} recent folds clear sharpe {p.gate_sharpe}")
    base_rate = float(ds.y.mean())
    base_brier = base_rate * (1.0 - base_rate)
    if holdout_report.brier > base_brier + 0.01:
        reasons.append(
            f"holdout brier {holdout_report.brier:.3f} worse than base rate {base_brier:.3f}"
        )

    report = GateReport(
        folds=fold_reports, holdout=holdout_report, passed=not reasons, reasons=reasons
    )
    logger.info(
        "Training gate evaluated",
        passed=report.passed,
        reasons=reasons,
        holdout_sharpe=round(holdout_report.portfolio.sharpe, 3),
        folds=len(fold_reports),
    )

    # Final model on the full history (fresh val split at the very end).
    final_model = _fit_on_dates(ds, sessions, p)
    if final_model is None:
        raise ValueError("full-history window is untrainable")
    return final_model, report
