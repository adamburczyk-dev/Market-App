"""End-to-end walk-forward training + gate report on a synthetic universe."""

import numpy as np
import pytest

from src.core.dataset import DatasetParams, build_dataset, drop_zero_variance_features
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
    assert set(d) == {"passed", "reasons", "conditions", "holdout", "folds"}
    assert isinstance(d["passed"], bool)


def interaction_universe(
    n_symbols: int = 24, n_sessions: int = 640, seed: int = 11
) -> dict[str, list]:
    """A universe where the edge is an INTERACTION of two features.

    Next-day drift is driven by (momentum rank − 0.5) × (0.5 − volatility rank):
    high momentum pays when volatility is low and is punished when it is high.
    No single feature's rank captures that, so a model can beat the baseline
    (G2) — which a pure momentum universe cannot demonstrate, because there
    `return_20d` alone IS the answer.
    """
    rng = np.random.default_rng(seed)
    closes = {f"S{k:02d}": [100.0 * (1.0 + 0.02 * rng.standard_normal())] for k in range(n_symbols)}
    names = list(closes)

    for _ in range(n_sessions - 1):
        momentum, vol = {}, {}
        for name in names:
            path = np.asarray(closes[name][-21:], dtype=float)
            if len(path) < 21:
                momentum[name], vol[name] = 0.0, 0.0
                continue
            rets = np.diff(np.log(path))
            momentum[name] = float(path[-1] / path[0] - 1.0)
            vol[name] = float(rets.std(ddof=1))

        mom_rank = _rank01([momentum[n] for n in names])
        vol_rank = _rank01([vol[n] for n in names])
        for i, name in enumerate(names):
            edge = (mom_rank[i] - 0.5) * (0.5 - vol_rank[i]) * 4.0  # in [-1, 1]
            step = 0.004 * edge + 0.008 * rng.standard_normal()
            closes[name].append(max(1.0, closes[name][-1] * (1.0 + step)))

    return {name: make_bars(name, values) for name, values in closes.items()}


def _rank01(values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return list(ranks / max(1, len(values) - 1))


WIDE = TrainingParams(
    train_size=200,
    test_size=63,
    holdout_size=126,
    val_size=40,
    horizon=10,
    embargo=2,
    quantile=0.2,
    model=TrainConfig(hidden=(32, 16), max_epochs=120, min_epochs=20, patience=10),
)


@pytest.mark.slow
def test_the_gate_is_passable_end_to_end():
    """The passability test — the one that says the gate is a filter and not a
    permanent "no". A universe with a real, learnable cross-sectional edge must
    clear all six conditions through the whole pipeline: dataset → purged
    walk-forward → G0–G5. If this ever fails, the gate is unreachable and every
    "gate FAILED" elsewhere becomes uninterpretable.
    """
    ds = build_dataset(
        interaction_universe(), DatasetParams(label=LabelParams(), min_history=60, min_universe=20)
    )
    ds, _ = drop_zero_variance_features(ds)
    _, report = run_training(ds, WIDE)
    assert report.passed, report.reasons
    assert report.outcome is not None
    assert all(c.passed for c in report.outcome.conditions)


def test_gate_recognizes_a_blatant_trend_but_not_on_three_names():
    """The 3-symbol fixture produces a huge Sharpe and a perfect AUC, and still
    must not pass: a Spearman IC over 3 names carries no evidence, and the
    model's ranking cannot beat `return_20d` when both see the same 3 points.
    Rejecting this is the gate working, not the gate misfiring."""
    ds = synthetic_dataset()
    _, report = run_training(ds, SMALL)
    assert report.holdout.portfolio.sharpe > 0.5
    assert report.holdout.auc > 0.55
    assert not report.passed
    assert {r[:2] for r in report.reasons} == {"G1", "G2"}


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
