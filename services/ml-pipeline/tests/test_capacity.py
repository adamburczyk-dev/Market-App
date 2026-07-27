"""The capacity probe must give opposite verdicts on the two cases that matter."""

import numpy as np
import pytest

from src.core.capacity import OVERFIT_CONFIG, run_capacity_probe
from src.core.dataset import Dataset
from src.core.model import TrainConfig

# Small enough to run in a test, still over-parameterized for the sample.
PROBE = TrainConfig(
    hidden=(128, 64), dropout=0.0, weight_decay=0.0, max_epochs=60, min_epochs=60, patience=10_000
)
PRODUCTION = TrainConfig(hidden=(32, 16), max_epochs=40, min_epochs=20, patience=10)


def make_dataset(x: np.ndarray, y: np.ndarray) -> Dataset:
    n = len(y)
    return Dataset(
        x=x,
        y=y,
        next_returns=np.zeros(n),
        dates=[],
        symbols=[],
        feature_names=[f"f{i}" for i in range(x.shape[1])],
    )


def learnable(n: int = 1200, seed: int = 5) -> Dataset:
    """Labels really are a function of the features (plus noise)."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, 6))
    score = x[:, 0] - x[:, 1] + 0.5 * x[:, 2] * x[:, 3]
    y = (score + rng.normal(0, 0.15, size=n) > score.mean()).astype(float)
    return make_dataset(x, y)


def unlearnable(n: int = 1200, seed: int = 5) -> Dataset:
    """Same shape, same base rate, labels independent of the features."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, 6))
    y = (rng.random(n) < 0.55).astype(float)
    return make_dataset(x, y)


def test_probe_finds_structure_when_there_is_some():
    probe = run_capacity_probe(learnable(), production_config=PRODUCTION, overfit_config=PROBE)
    assert probe.auc_train_real > probe.auc_train_shuffled + 0.05
    assert probe.gap > 0.05
    assert "EXISTS" in probe.verdict


def test_probe_calls_memorization_what_it_is():
    """The control is the whole point: on random labels the big model still
    fits the TRAINING rows (that is memorization), so a high train AUC alone
    would be read as success. The gap against shuffled labels is what decides.
    """
    probe = run_capacity_probe(unlearnable(), production_config=PRODUCTION, overfit_config=PROBE)
    assert probe.gap < 0.05
    assert "NO learnable structure" in probe.verdict


def test_probe_refuses_degenerate_input():
    tiny = make_dataset(np.random.default_rng(0).random((50, 4)), np.zeros(50))
    with pytest.raises(ValueError, match="200 rows"):
        run_capacity_probe(tiny)

    one_class = make_dataset(np.random.default_rng(0).random((300, 4)), np.ones(300))
    with pytest.raises(ValueError, match="both classes"):
        run_capacity_probe(one_class, production_config=PRODUCTION, overfit_config=PROBE)


def test_default_overfit_config_has_regularization_off():
    # If any of these drifts back on, the probe stops answering its question.
    assert OVERFIT_CONFIG.dropout == 0.0
    assert OVERFIT_CONFIG.weight_decay == 0.0
    assert OVERFIT_CONFIG.min_epochs == OVERFIT_CONFIG.max_epochs  # early stopping cannot fire
