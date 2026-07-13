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
    return build_dataset(universe, DatasetParams(label=LabelParams(), min_history=60))


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
    ds = build_dataset(universe, DatasetParams(label=LabelParams(), min_history=60))
    assert ds.n_samples > 0
    _, report = run_training(ds, SMALL)
    assert not report.passed, "pure noise cleared the activation gate"
    assert report.reasons
