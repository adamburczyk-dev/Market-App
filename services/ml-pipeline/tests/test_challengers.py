"""The GBDT challenger and seed ensembling (P4-1, P4-2).

Both are measured through the SAME walk-forward and the SAME gate as the MLP,
so what is being compared is the model class rather than the setup. Two things
have to be pinned: the challenger must be able to learn something the MLP path
can also be scored on, and it must NOT be able to reach the registry — the
store persists an MLP state_dict, so logging a booster would write an artifact
that `load()` fails on later.
"""

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from src.core.ensemble import EnsembleModel, train_ensemble
from src.core.gbdt import GbdtConfig, feature_gain, train_gbdt
from src.core.model import TrainConfig, TrainedModel, train_classifier
from src.core.training import run_training

RNG = np.random.default_rng(11)
D0 = datetime(2023, 1, 2, tzinfo=UTC)


def learnable(n: int = 4000, n_features: int = 6, noise: float = 1.0):
    """A dataset with real, non-linear structure: the label depends on the
    INTERACTION of two columns, which is what trees are supposed to be good at
    and a linear read of any single feature cannot recover."""
    x = RNG.random((n, n_features))
    signal = (x[:, 0] - 0.5) * (x[:, 1] - 0.5) * 4.0
    y = (signal + RNG.normal(0, noise, n) > 0).astype(float)
    return x, y


def split(x, y, frac: float = 0.75):
    cut = int(len(y) * frac)
    return x[:cut], y[:cut], x[cut:], y[cut:]


def auc(y_true, scores) -> float:
    from src.core.evaluation import auc as _auc

    return _auc(np.asarray(y_true), np.asarray(scores))


FEATURES = [f"f{i}" for i in range(6)]


# --- P4-1: the GBDT challenger --------------------------------------------


def test_gbdt_learns_structure_and_reports_a_calibrated_probability():
    x, y = learnable(noise=0.5)
    xt, yt, xv, yv = split(x, y)
    model = train_gbdt(xt, yt, xv, yv, FEATURES)

    probs = model.predict_proba(xv)
    assert probs.shape == (len(yv),)
    assert np.all((probs >= 0) & (probs <= 1))
    assert auc(yv, probs) > 0.65, "the interaction should be recoverable by trees"
    assert model.temperature > 0


def test_gbdt_finds_nothing_in_noise_and_says_so():
    """The failure mode that matters: on labels with no structure the booster
    must not manufacture separation out of sample."""
    x = RNG.random((3000, 6))
    y = (RNG.random(3000) > 0.5).astype(float)
    xt, yt, xv, yv = split(x, y)
    model = train_gbdt(xt, yt, xv, yv, FEATURES)
    assert 0.42 < auc(yv, model.predict_proba(xv)) < 0.58


def test_gbdt_diagnostics_answer_the_questions_g0_asks():
    x, y = learnable(noise=0.5)
    xt, yt, xv, yv = split(x, y)
    d = train_gbdt(xt, yt, xv, yv, FEATURES).diagnostics
    assert d["model_kind"] == "gbdt"
    # G0 reads best_epoch: a booster stopping at its first tree learned nothing,
    # exactly like an MLP whose validation loss never improved.
    assert d["best_epoch"] > 1
    assert d["pred_std_pre_calibration"] > 0
    assert 0 < d["n_features_used"] <= len(FEATURES)


def test_gbdt_honours_sample_weights():
    """P0-3's uniqueness weights must reach the booster, not be ignored: a row
    weighted to zero cannot influence the fit."""
    x, y = learnable(n=2000, noise=0.5)
    xt, yt, xv, yv = split(x, y)
    weights = np.ones(len(yt))
    weights[: len(yt) // 2] = 0.0
    weighted = train_gbdt(xt, yt, xv, yv, FEATURES, sample_weights=weights)
    unweighted = train_gbdt(xt, yt, xv, yv, FEATURES)
    assert not np.allclose(weighted.predict_proba(xv), unweighted.predict_proba(xv))

    with pytest.raises(ValueError, match="sample_weights"):
        train_gbdt(xt, yt, xv, yv, FEATURES, sample_weights=np.ones(3))


def test_feature_gain_sums_to_one_over_used_features():
    x, y = learnable(noise=0.5)
    xt, yt, xv, yv = split(x, y)
    gains = feature_gain(train_gbdt(xt, yt, xv, yv, FEATURES))
    assert set(gains) == set(FEATURES)
    assert sum(gains.values()) == pytest.approx(1.0, abs=0.01)


def test_gbdt_is_deterministic():
    """A training run has to be reproducible or none of the diagnostics mean
    anything across reruns (hence n_jobs=1, not speed)."""
    x, y = learnable(n=1500, noise=0.5)
    xt, yt, xv, yv = split(x, y)
    cfg = GbdtConfig(n_estimators=80)
    a = train_gbdt(xt, yt, xv, yv, FEATURES, cfg).predict_proba(xv)
    b = train_gbdt(xt, yt, xv, yv, FEATURES, cfg).predict_proba(xv)
    assert np.allclose(a, b)


# --- P4-2: seed ensembling -------------------------------------------------


def test_ensemble_averages_members_and_reports_their_disagreement():
    x, y = learnable(n=1200, noise=1.0)
    xt, yt, xv, yv = split(x, y)

    def fit(seed: int):
        return train_classifier(
            xt, yt, xv, yv, FEATURES, TrainConfig(max_epochs=40, min_epochs=5, seed=seed)
        )

    ensemble = train_ensemble(fit, seeds=[1, 2, 3], reference_x=xv)
    assert isinstance(ensemble, EnsembleModel)
    assert ensemble.diagnostics["n_members"] == 3
    # the averaged prediction is exactly the mean of the members
    members = ensemble.member_predictions(xv)
    assert np.allclose(ensemble.predict_proba(xv), members.mean(axis=0))
    # disagreement is measured, not assumed away
    assert ensemble.diagnostics["seed_disagreement"] >= 0
    assert (
        ensemble.diagnostics["pred_std_ensemble"]
        <= ensemble.diagnostics["pred_std_member_mean"] + 1e-9
    ), "averaging cannot increase the spread of the score"


def test_a_single_seed_is_not_dressed_up_as_an_ensemble():
    """An 'ensemble' of one would report a disagreement of exactly zero, which
    reads as perfect agreement rather than as no measurement."""
    x, y = learnable(n=800, noise=1.0)
    xt, yt, xv, yv = split(x, y)

    def fit(seed: int):
        return train_classifier(
            xt, yt, xv, yv, FEATURES, TrainConfig(max_epochs=20, min_epochs=5, seed=seed)
        )

    single = train_ensemble(fit, seeds=[5], reference_x=xv)
    assert isinstance(single, TrainedModel)
    assert "seed_disagreement" not in single.diagnostics


def test_ensemble_best_epoch_is_the_minimum_so_g0_still_bites():
    """G0 rejects a model whose fit never improved. If the ensemble reported the
    MEAN or MAX best_epoch, one member restored at epoch 1 would be hidden by
    the others — exactly the failure run #2 had, twice."""

    class Stub:
        def __init__(self, best_epoch: int) -> None:
            self.feature_names = FEATURES
            self.diagnostics = {"best_epoch": best_epoch, "epochs_run": 30}

        def predict_proba(self, x):
            return np.full(len(x), 0.5)

    members = {1: Stub(1), 2: Stub(20), 3: Stub(25)}
    ensemble = train_ensemble(lambda s: members[s], seeds=[1, 2, 3])
    assert ensemble.diagnostics["best_epoch"] == 1


def test_untrainable_window_yields_nothing_rather_than_an_empty_ensemble():
    assert train_ensemble(lambda _: None, seeds=[1, 2, 3]) is None


# --- both classes run through the SAME walk-forward and gate ---------------


def test_the_challenger_produces_a_comparable_report():
    """The point of P4-1: an identical evaluation, so what differs between the
    two reports is the model class and not the setup around it."""
    from .test_training import SMALL, synthetic_dataset

    ds = synthetic_dataset()
    mlp_model, mlp_report = run_training(ds, SMALL)
    gbdt_model, gbdt_report = run_training(
        ds, replace(SMALL, model_kind="gbdt", gbdt=GbdtConfig(n_estimators=60, max_depth=3))
    )

    assert set(mlp_report.as_dict()) == set(gbdt_report.as_dict())
    assert len(gbdt_report.folds) == len(mlp_report.folds)
    assert gbdt_report.holdout.n_test == mlp_report.holdout.n_test
    assert gbdt_model.feature_names == mlp_model.feature_names
    assert gbdt_model.diagnostics["model_kind"] == "gbdt"
    # ...and the challenger is NOT an MLP, which is what stops it reaching the
    # registry (the store persists a state_dict it could not reconstruct).
    assert not isinstance(gbdt_model, TrainedModel)
    assert isinstance(mlp_model, TrainedModel)


def test_seed_ensembling_runs_through_training_and_is_reported():
    from .test_training import SMALL, synthetic_dataset

    ds = synthetic_dataset()
    model, report = run_training(ds, replace(SMALL, n_seeds=3))
    assert model.diagnostics["n_members"] == 3
    assert "seed_disagreement" in model.diagnostics
    # the fold reports still carry the per-window fit diagnostics G0 reads
    assert report.holdout.fit.get("best_epoch") is not None
