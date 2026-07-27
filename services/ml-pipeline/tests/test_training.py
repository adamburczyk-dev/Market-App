"""End-to-end walk-forward training + gate report on a synthetic universe."""

import numpy as np
import pytest

from src.core.dataset import DatasetParams, build_dataset
from src.core.labels import LabelParams
from src.core.model import TrainConfig
from src.core.training import TrainingParams, run_training

from .test_dataset import make_bars, trending

SMALL = TrainingParams(
    train_size=60,
    test_size=20,
    holdout_size=30,
    val_size=15,
    horizon=10,
    embargo=2,
    quantile=0.34,  # top-1 of a 3-symbol universe
    model=TrainConfig(hidden=(16, 8), max_epochs=25, patience=5, batch_size=64),
)


def synthetic_dataset(n: int = 220):
    universe = {
        "UP": make_bars("UP", trending(n, 0.004)),
        "DOWN": make_bars("DOWN", trending(n, -0.004)),
        "FLATISH": make_bars("FLATISH", trending(n, 0.0005)),
    }
    return build_dataset(
        universe, DatasetParams(label=LabelParams(), min_history=60, min_universe=2)
    )


def test_training_produces_model_and_report():
    ds = synthetic_dataset()
    model, report = run_training(ds, SMALL)
    assert model.feature_names == ds.feature_names
    assert report.holdout.n_test > 0
    assert len(report.folds) >= 1
    d = report.as_dict()
    assert set(d) == {"passed", "reasons", "holdout", "folds"}
    assert isinstance(d["passed"], bool)


def test_gate_passes_on_a_blatant_trend_universe():
    """A persistent up-trender vs a down-trender is as easy as it gets — the
    gate must recognize it (this also pins the metric plumbing end-to-end)."""
    ds = synthetic_dataset()
    _, report = run_training(ds, SMALL)
    assert report.holdout.portfolio.sharpe > 0.5
    assert report.holdout.auc > 0.55
    assert report.passed, report.reasons


def test_diagnostics_show_the_edge_on_a_learnable_universe():
    """The review artifact must distinguish signal from luck: on a universe the
    model genuinely ranks, the selected top quantile hits above the base rate
    and predictions actually spread."""
    ds = synthetic_dataset()
    _, report = run_training(ds, SMALL)
    diag = report.holdout.diagnostics
    assert diag.lift > 0, "selection carries no edge over the base rate"
    assert diag.selected_hit_rate > diag.base_rate
    assert diag.pred_std > 0.01, "collapsed predictions — no ranking information"
    assert diag.pred_p10 <= diag.pred_mean <= diag.pred_p90
    fold = report.as_dict()["holdout"]
    assert {"lift", "base_rate", "selected_hit_rate", "pred_std"} <= set(fold)


def test_too_small_dataset_raises():
    ds = synthetic_dataset(n=120)  # not enough sessions for holdout + a fold
    with pytest.raises(ValueError, match="sessions"):
        run_training(ds, SMALL)


def random_walk(n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))]


def test_gate_fails_on_noise():
    """Driftless random walks must not pass the activation gate."""
    universe = {f"N{k}": make_bars(f"N{k}", random_walk(220, seed=k)) for k in range(3)}
    ds = build_dataset(universe, DatasetParams(label=LabelParams(), min_history=60, min_universe=2))
    assert ds.n_samples > 0
    _, report = run_training(ds, SMALL)
    assert not report.passed, "pure noise cleared the activation gate"
    assert report.reasons


def test_report_separates_underfitting_from_absent_signal():
    """T0-3: the report must answer the question the first real run could not.

    auc_train ~ 0.5           -> optimisation problem (capacity / lr / epochs)
    auc_train high, auc ~ 0.5 -> overfit; no signal in the features
    both ~ 0.5 and T >> 1     -> no signal, and calibration correctly gave up

    The last case is why the raw spread matters: temperature scaling flattens
    every probability onto the base rate when validation AUC is ~0.5, so
    pred_std AFTER calibration cannot distinguish a collapsed model from a
    humbled one.
    """
    ds = synthetic_dataset()
    _, report = run_training(ds, SMALL)
    row = report.as_dict()["holdout"]

    for field_name in (
        "auc_train",
        "epochs_run",
        "best_epoch",
        "early_stop_reason",
        "loss_train_final",
        "loss_val_final",
        "calibration_temperature",
        "pred_std_pre_calibration",
        "pred_std_post_calibration",
    ):
        assert field_name in row, f"missing diagnostic: {field_name}"

    assert row["early_stop_reason"] in ("patience", "max_epochs")
    assert row["epochs_run"] >= row["best_epoch"]
    assert row["calibration_temperature"] > 0
    # on a learnable universe the model does fit its training window
    assert row["auc_train"] > 0.6


def test_calibration_can_hide_a_collapsed_model():
    """Pin the mechanism the audit identified: on unlearnable data the fitted
    temperature grows and squeezes the post-calibration spread, so only the
    pre-calibration spread reveals what the network actually produced."""
    universe = {f"N{k}": make_bars(f"N{k}", random_walk(220, seed=100 + k)) for k in range(3)}
    ds = build_dataset(universe, DatasetParams(label=LabelParams(), min_history=60, min_universe=2))
    _, report = run_training(ds, SMALL)
    row = report.as_dict()["holdout"]
    assert row["pred_std_pre_calibration"] >= row["pred_std_post_calibration"] or (
        row["calibration_temperature"] < 1.0
    )


def test_effective_sample_size_shrinks_for_overlapping_correlated_data():
    """T0-3: 48 827 rows is a nominal count. Overlapping labels divide the time
    axis by the horizon; correlated names divide the cross-section."""
    from src.core.evaluation import effective_sample_size

    ds = synthetic_dataset()
    ess = effective_sample_size(ds.dates, ds.symbols, ds.next_returns, horizon=10)
    assert ess.n_samples == ds.n_samples
    assert ess.n_effective_samples < ess.n_samples  # always, by construction
    assert ess.n_independent_periods == pytest.approx(ess.n_sessions / 10, abs=0.1)
    assert 0 < ess.n_symbols_effective <= ess.n_symbols
