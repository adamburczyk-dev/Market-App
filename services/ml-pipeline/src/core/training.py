"""Purged walk-forward training + the activation-gate report (plan §6–§7).

The holdout — the most recent ``holdout_size`` sessions — is never touched
during model selection. Walk-forward folds run over the remaining history;
each fold trains on its purged window (with an internal, also-purged
train/val split for early stopping + calibration) and is scored on its test
block. The gate itself lives in ``core/gate.py`` — six conditions (sanity,
rank information, vs baseline, economics, calibration, deflated Sharpe), each
closing a different way a model with no edge can look profitable. Only a
gate-passing model may serve non-HOLD signals.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog

from src.core.dataset import Dataset
from src.core.ensemble import Predictor, train_ensemble
from src.core.evaluation import (
    PortfolioResult,
    RelativeMetrics,
    SelectionDiagnostics,
    auc,
    brier,
    effective_sample_size,
    per_feature_ic,
    relative_metrics,
    selection_diagnostics,
    top_quantile_portfolio,
)
from src.core.gate import GateOutcome, GateThresholds, evaluate_gate
from src.core.gbdt import GbdtConfig, train_gbdt
from src.core.importance import (
    NOISE_FEATURE,
    ImportanceReport,
    noise_control_column,
    permutation_importance,
)
from src.core.labels import LABEL_HORIZON
from src.core.model import TrainConfig, train_classifier
from src.core.splits import purged_walk_forward

logger = structlog.get_logger()


@dataclass(frozen=True)
class TrainingParams:
    train_size: int = 756  # ~3y of sessions
    # Sized against the LABEL HORIZON, not the calendar. The tranche book
    # refreshes sleeve `t mod horizon`, so it only reaches steady state after
    # `horizon` sessions: at test_size == horizon the book is in warm-up for
    # 100% of every fold and every Sharpe, turnover and cost figure is a
    # transient. 3x the horizon leaves ~67% of each fold in steady state.
    # Arithmetic and the rejected alternative (pre-seeding the sleeves, which
    # would change the construction and break comparability with runs #1-#3)
    # are in docs/decisions/06.
    test_size: int = 189  # 3x horizon
    holdout_size: int = 252  # 4x horizon (~1y), untouched during selection
    # One horizon of validation is ONE independent label episode — too little to
    # early-stop or calibrate on, and that mechanism failing is what produced
    # run #3's collapse to a constant.
    val_size: int = 126  # 2x horizon
    horizon: int = LABEL_HORIZON  # label horizon → purge width
    embargo: int = 5
    quantile: float = 0.2
    cost_bps: float = 5.0
    gate_sharpe: float = 0.5
    # T0-4: hold a position for the label horizon instead of rebalancing the
    # whole book daily, so the evaluated object matches the trained one.
    overlapping_tranches: bool = True
    model: TrainConfig = field(default_factory=TrainConfig)
    # P4-1: which model class to fit. "gbdt" is a CHALLENGER — evaluated through
    # the identical walk-forward and gate, but not registrable (the store
    # persists an MLP state_dict), so a comparison cannot be quietly promoted.
    model_kind: str = "mlp"
    gbdt: GbdtConfig = field(default_factory=GbdtConfig)
    # P4-2: how many seeds to average. Run #2's folds swung from Sharpe -1.61 to
    # +4.54 on identical data; part of that is the initialisation, and averaging
    # removes exactly that part. 1 = no ensembling (the previous behaviour).
    n_seeds: int = 1
    # T1-3: the six-condition gate. `gate_sharpe` above feeds its economics
    # condition, so the project rule (OOS Sharpe > 0.5) stays in one place.
    gate: GateThresholds = field(default_factory=GateThresholds)
    # Faza 3: permutation repeats for the holdout importance table. 0 skips it.
    # It costs (features + families) x repeats forward passes on the holdout —
    # no fitting — so the default is on: a training run that cannot say which
    # inputs its model used is a run nobody can act on.
    importance_repeats: int = 3


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
    # E1: how much worse (positive) or better (negative) the probabilities are
    # than predicting the window's base rate every day, and the standard error
    # of that paired difference. A gate that demands a strict improvement fails
    # a genuinely discriminating model on rounding noise; one that ignores the
    # sign lets a miscalibrated model through. The SE is what separates them.
    brier_delta: float = 0.0
    brier_delta_se: float = 0.0
    fit: dict[str, Any] = field(default_factory=dict)
    # T0-5: benchmark-relative and rank-based measurement. A long-only Sharpe
    # in a window where most names rose says more about the market than the
    # model; IC/ICIR and the long-short leg do not have that failure mode.
    relative: RelativeMetrics | None = None
    baseline_ic: dict[str, float] = field(default_factory=dict)
    # P2-1: the same features with their IC t-statistics — the stage-2 gate
    # ("a family enters only if it raises the evidence") is read off this.
    feature_ic: dict[str, Any] = field(default_factory=dict)
    # P0-4: what this particular window was made of — label mix, independent
    # information, cross-section width. A fold's metrics are not comparable to
    # another's without it.
    window: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateReport:
    folds: list[FoldReport]
    holdout: FoldReport
    passed: bool
    reasons: list[str]
    outcome: GateOutcome | None = None  # per-condition detail (G0–G5)
    # Faza 3: which inputs the HOLDOUT model actually used, measured by
    # within-session permutation on the holdout itself (core/importance.py).
    # Attached to the report rather than to `holdout` because it is a property
    # of one model on untouched data, not a per-window score.
    importance: ImportanceReport | None = None

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            # G0–G5 with their numbers: a failed gate must say WHICH question
            # was answered "no", and a passed one must show what it cleared.
            "conditions": self.outcome.as_dict() if self.outcome is not None else [],
            # Which inputs the model used, not which columns correlate. `null`
            # means it was not measured (repeats set to 0, or a holdout too
            # short to pair) — never "no feature mattered".
            "importance": self.importance.as_dict() if self.importance is not None else None,
            "holdout": fold_as_dict(self.holdout),
            "folds": [fold_as_dict(f) for f in self.folds],
        }


def fold_as_dict(f: FoldReport) -> dict:
    """One window's numbers, flattened.

    Module-level rather than a closure inside ``GateReport.as_dict`` because
    the progress checkpoint serializes folds too, and a checkpoint whose folds
    had a different shape from the final report's would be unreadable next to
    it — the whole point of the checkpoint is that it can be compared with what
    a finished run would have said.
    """
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
        "brier_delta": round(f.brier_delta, 5),
        "brier_delta_se": round(f.brier_delta_se, 5),
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
        "feature_ic": f.feature_ic,
        **f.window,
    }


def _mask(dates: list[datetime], allowed: set[datetime]) -> np.ndarray:
    return np.array([d in allowed for d in dates], dtype=bool)


def holdout_split(
    ds: Dataset, p: TrainingParams
) -> tuple[list[datetime], list[datetime], list[datetime]]:
    """(work sessions, purged train sessions, holdout sessions).

    One definition for the two callers that need the seam — the training run
    and the importance study. A study that measured on a holdout drawn a few
    sessions differently from the one the gate scored would be reporting on a
    model fitted to data the gate's model never saw, and nothing in either
    report would say so.
    """
    sessions = sorted(set(ds.dates))
    if len(sessions) <= p.holdout_size:
        raise ValueError(
            f"dataset has {len(sessions)} sessions; needs more than the "
            f"{p.holdout_size}-session holdout"
        )
    work = sessions[: -p.holdout_size]
    holdout = sessions[-p.holdout_size :]
    gap = p.horizon + p.embargo
    return work, (work[:-gap] if gap else work), holdout


def fit_on_dates(ds: Dataset, dates: list[datetime], params: TrainingParams) -> Predictor | None:
    """Train with the window's tail as the (purged) validation fold.

    Dispatches on `model_kind` and averages over `n_seeds` — both paths get the
    identical fit/validation split and the identical uniqueness weights, so the
    comparison between them is about the model class, not about the setup.
    """
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
    # P0-3: overlapping labels are one episode, not `horizon` episodes.
    weights = ds.weights[fit_mask] if len(ds.weights) == len(ds.dates) else None

    def fit_one(seed: int) -> Predictor | None:
        if params.model_kind == "gbdt":
            return train_gbdt(
                ds.x[fit_mask],
                ds.y[fit_mask],
                ds.x[val_mask],
                ds.y[val_mask],
                ds.feature_names,
                replace(params.gbdt, seed=seed),
                sample_weights=weights,
            )
        return train_classifier(
            ds.x[fit_mask],
            ds.y[fit_mask],
            ds.x[val_mask],
            ds.y[val_mask],
            ds.feature_names,
            replace(params.model, seed=seed),
            sample_weights=weights,
        )

    base_seed = params.gbdt.seed if params.model_kind == "gbdt" else params.model.seed
    seeds = [base_seed + i for i in range(max(1, params.n_seeds))]
    return train_ensemble(fit_one, seeds, reference_x=ds.x[val_mask])


def gate_outcome(holdout: FoldReport, folds: list[FoldReport], p: TrainingParams) -> GateOutcome:
    """Apply the G0–G5 gate with this run's thresholds (see core/gate.py).

    Separate from ``run_training`` so the decision can be tested on numbers
    rather than on a model that has to be lucky twice.
    """
    thresholds = replace(p.gate, sharpe=p.gate_sharpe)
    return evaluate_gate(holdout, folds, thresholds)


def _window_stats(ds: Dataset, mask: np.ndarray, params: TrainingParams) -> dict[str, Any]:
    """Label mix and independent information of one evaluation window (P0-4)."""
    n = int(mask.sum())
    if n == 0:
        return {}
    stats: dict[str, Any] = {"window_rows": n}
    if len(ds.barriers) == len(ds.dates):
        counts: dict[str, int] = {}
        for barrier, keep in zip(ds.barriers, mask, strict=True):
            if keep:
                counts[barrier] = counts.get(barrier, 0) + 1
        stats["window_label_resolution"] = counts
        stats["window_vertical_share"] = round(counts.get("vertical", 0) / n, 4)
    if len(ds.weights) == len(ds.dates):
        stats["window_weighted_rows"] = round(float(ds.weights[mask].sum()), 1)
    ess = effective_sample_size(
        [d for d, m in zip(ds.dates, mask, strict=True) if m],
        [s for s, m in zip(ds.symbols, mask, strict=True) if m],
        ds.next_returns[mask],
        params.horizon,
    )
    stats["window_effective_samples"] = round(ess.n_effective_samples, 1)
    return stats


def score_window(
    ds: Dataset,
    model: Predictor,
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
        tranches=params.horizon if params.overlapping_tranches else 1,
    )
    relative = relative_metrics(
        dates,
        [s for s, m in zip(ds.symbols, mask, strict=True) if m],
        probs,
        ds.next_returns[mask],
        quantile=params.quantile,
        cost_bps=params.cost_bps,
        # same book the gate scores — otherwise `sharpe` and `sharpe_net`
        # describe two different portfolios in one table
        tranches=params.horizon if params.overlapping_tranches else 1,
    )
    # The rule: a model that cannot beat the raw rank of one feature has not
    # earned the ML layer. return_20d is the strongest single candidate here.
    window_stats = _window_stats(ds, mask, params)
    y_true = ds.y[mask]
    base_rate = float(y_true.mean()) if len(y_true) else 0.5
    paired = (probs - y_true) ** 2 - (base_rate - y_true) ** 2
    brier_delta = float(paired.mean()) if len(paired) else 0.0
    brier_delta_se = float(paired.std(ddof=1) / np.sqrt(len(paired))) if len(paired) > 1 else 0.0
    # Every feature's standalone IC, not just one declared yardstick (P2-1).
    # G2 takes the MAX of these, so adding a feature can only make the gate
    # harder — the right direction for a comparator selected on the same window.
    feature_ic = per_feature_ic(dates, ds.x[mask], ds.feature_names, ds.next_returns[mask])
    baseline_ic = {name: f.mean for name, f in feature_ic.items()}

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
        feature_ic={name: f.as_dict() for name, f in feature_ic.items()},
        window=window_stats,
        brier_delta=brier_delta,
        brier_delta_se=brier_delta_se,
    )


def _holdout_importance(
    ds: Dataset, model: Predictor, holdout: list[datetime], p: TrainingParams
) -> ImportanceReport | None:
    """Permutation importance of the holdout model, on the holdout (Faza 3).

    Degrades to ``None`` rather than failing the training run: a holdout too
    short to pair session-by-session is a reason to omit the table, not a
    reason to throw away a completed walk-forward. The report renders `null` as
    "not measured", which is a different statement from "no feature mattered".
    """
    if p.importance_repeats <= 0:
        return None
    mask = _mask(ds.dates, set(holdout))
    try:
        return permutation_importance(
            model,
            ds.x[mask],
            ds.y[mask],
            [d for d, keep in zip(ds.dates, mask, strict=True) if keep],
            ds.next_returns[mask],
            ds.feature_names,
            n_repeats=p.importance_repeats,
        )
    except ValueError as exc:
        logger.warning("Feature importance not measured", reason=str(exc))
        return None


def run_importance_study(
    ds: Dataset,
    params: TrainingParams | None = None,
    n_repeats: int = 5,
    seed: int = 11,
) -> dict[str, Any]:
    """Importance with a planted noise column — the empirical floor of the table.

    The training run measures the PRODUCTION model, so its feature contract is
    the production contract and nothing may be added to it. This study fits its
    own model through the identical path with one extra column that is a valid
    cross-sectional rank and predicts nothing by construction, and reports what
    that column scores. That number is the honest bar: the Šidák correction
    assumes the tests are independent and the model is indifferent to a useless
    input, and a model that latched onto the noise says so here instead of in a
    footnote.

    Diagnostic only — the fitted model carries a column serving cannot produce,
    so it is neither registered nor registrable.
    """
    p = params or TrainingParams()
    work, train_dates, holdout = holdout_split(ds, p)
    planted = replace(
        ds,
        x=np.column_stack([ds.x, noise_control_column(ds.dates, seed=seed)]),
        feature_names=[*ds.feature_names, NOISE_FEATURE],
    )
    model = fit_on_dates(planted, train_dates, p)
    if model is None:
        raise ValueError("window before the holdout is untrainable (too small or single-class)")
    mask = _mask(planted.dates, set(holdout))
    report = permutation_importance(
        model,
        planted.x[mask],
        planted.y[mask],
        [d for d, keep in zip(planted.dates, mask, strict=True) if keep],
        planted.next_returns[mask],
        planted.feature_names,
        n_repeats=n_repeats,
        seed=seed,
    )
    logger.info(
        "Importance study finished",
        holdout_sessions=len(holdout),
        train_sessions=len(train_dates),
        features=len(planted.feature_names),
    )
    return {
        "sessions": len(work) + len(holdout),
        "train_sessions": len(train_dates),
        "holdout_sessions": len(holdout),
        "model_kind": model.diagnostics.get("model_kind", "mlp"),
        # Said out loud because the table below contains a column that does not
        # exist in the model the system serves.
        "noise_control_planted": True,
        "registrable": False,
        "importance": report.as_dict(),
    }


ProgressCallback = Callable[[dict[str, Any]], None]


def report_progress(
    callback: ProgressCallback | None,
    stage: str,
    done: int,
    total: int,
    folds: list[FoldReport],
) -> None:
    """Publish what the run has finished so far. Never raises.

    A checkpoint exists to survive a failure, so it must not be able to CAUSE
    one: a callback that throws (a full disk, a read-only mount) would abort a
    training pass in order to protect a diagnostic, which is backwards.

    The folds are serialized with the same function the final report uses, so
    an interrupted run and a finished one can be read side by side. `pred_std`
    is the field that makes this worth writing at all — a pass killed at fold
    40 still answers whether the model was varying or had already collapsed,
    and those two outcomes call for opposite next moves.
    """
    if callback is None:
        return
    try:
        callback(
            {
                "stage": stage,
                "folds_done": done,
                "folds_total": total,
                "updated_at": datetime.now(UTC).isoformat(),
                "folds": [fold_as_dict(f) for f in folds],
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Progress callback failed", stage=stage, error=str(exc))


def run_training(
    ds: Dataset,
    params: TrainingParams | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[Predictor, GateReport]:
    """Walk-forward evaluation → gate report → final model.

    The returned model is trained on ALL available history (with the standard
    internal val split) regardless of the gate outcome — the caller decides
    what a failed gate means (register as non-production, keep serving HOLD).
    Raises ValueError when the dataset is too small to evaluate at all.

    ``on_progress`` is called after every completed stage. On the full universe
    this pass runs for hours and, until it returns, exists nowhere: an
    interruption at fold 40 of 61 destroyed 40 fits that had already been paid
    for AND the evidence of what they showed. The callback is the only channel
    through which a run that never finishes can still say something — notably
    whether predictions were varying, which is the difference between "the
    model was learning and we lost it" and "it had already collapsed".
    """
    p = params or TrainingParams()
    sessions = sorted(set(ds.dates))
    if len(sessions) < p.holdout_size + p.train_size + p.test_size:
        raise ValueError(
            f"dataset has {len(sessions)} sessions; needs ≥ "
            f"{p.holdout_size + p.train_size + p.test_size} for holdout + one fold"
        )

    work, holdout_train, holdout = holdout_split(ds, p)

    folds = purged_walk_forward(work, p.train_size, p.test_size, p.horizon, p.embargo)
    report_progress(on_progress, "folds", 0, len(folds), [])
    fold_reports: list[FoldReport] = []
    for k, fold in enumerate(folds):
        model = fit_on_dates(ds, list(fold.train_dates), p)
        if model is None:
            logger.warning("Fold skipped — untrainable window", fold=k)
            report_progress(on_progress, "folds", k + 1, len(folds), fold_reports)
            continue
        fold_reports.append(
            score_window(
                ds,
                model,
                set(fold.test_dates),
                f"fold_{k}",
                len(fold.train_dates),
                p,
                fit_dates=set(fold.train_dates),
            )
        )
        report_progress(on_progress, "folds", k + 1, len(folds), fold_reports)

    # Holdout model: trained on everything BEFORE the holdout, purged at the seam.
    report_progress(on_progress, "holdout", len(folds), len(folds), fold_reports)
    holdout_model = fit_on_dates(ds, holdout_train, p)
    if holdout_model is None:
        raise ValueError("holdout window is untrainable (too small or single-class)")
    holdout_report = score_window(
        ds,
        holdout_model,
        set(holdout),
        "holdout",
        len(holdout_train),
        p,
        fit_dates=set(holdout_train),
    )

    outcome = gate_outcome(holdout_report, fold_reports, p)
    reasons = outcome.reasons

    report = GateReport(
        folds=fold_reports,
        holdout=holdout_report,
        passed=outcome.passed,
        reasons=reasons,
        outcome=outcome,
        importance=_holdout_importance(ds, holdout_model, holdout, p),
    )
    logger.info(
        "Training gate evaluated",
        passed=report.passed,
        reasons=reasons,
        holdout_sharpe=round(holdout_report.portfolio.sharpe, 3),
        folds=len(fold_reports),
    )

    # Final model on the full history (fresh val split at the very end).
    # Checkpointed because everything scored is already done by here: an
    # interruption during this last fit loses a model, not the evidence.
    report_progress(on_progress, "final_model", len(folds), len(folds), fold_reports)
    final_model = fit_on_dates(ds, sessions, p)
    if final_model is None:
        raise ValueError("full-history window is untrainable")
    return final_model, report
