"""Tests for the MLP classifier + temperature calibration."""

import numpy as np

from src.core.evaluation import auc
from src.core.model import TrainConfig, train_classifier

FAST = TrainConfig(hidden=(16, 8), max_epochs=80, min_epochs=30, patience=12, batch_size=64, seed=7)


def separable(n: int = 600, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Rank-space features where feature 0 drives the label with noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 4))
    p = 0.15 + 0.7 * x[:, 0]
    y = (rng.uniform(size=n) < p).astype(float)
    return x, y


def test_learns_a_separable_signal():
    x, y = separable()
    model = train_classifier(x[:400], y[:400], x[400:500], y[400:500], ["a", "b", "c", "d"], FAST)
    probs = model.predict_proba(x[500:])
    assert auc(y[500:], probs) > 0.65  # far above chance on held-out rows
    assert np.all((probs >= 0) & (probs <= 1))


def test_probability_moves_with_the_driving_feature():
    x, y = separable()
    model = train_classifier(x[:400], y[:400], x[400:500], y[400:500], ["a", "b", "c", "d"], FAST)
    low = np.array([[0.05, 0.5, 0.5, 0.5]])
    high = np.array([[0.95, 0.5, 0.5, 0.5]])
    assert model.predict_proba(high)[0] > model.predict_proba(low)[0]


def test_deterministic_given_seed():
    x, y = separable()
    a = train_classifier(x[:300], y[:300], x[300:400], y[300:400], ["a", "b", "c", "d"], FAST)
    b = train_classifier(x[:300], y[:300], x[300:400], y[300:400], ["a", "b", "c", "d"], FAST)
    probe = x[400:420]
    assert np.allclose(a.predict_proba(probe), b.predict_proba(probe))


def test_temperature_is_sane_and_recorded():
    x, y = separable()
    model = train_classifier(x[:400], y[:400], x[400:500], y[400:500], ["a", "b", "c", "d"], FAST)
    assert 0.05 <= model.temperature <= 20.0
    assert model.history["epochs"] >= 1
    assert model.feature_names == ["a", "b", "c", "d"]


def test_single_class_training_does_not_crash():
    # degenerate but shouldn't raise: pos_weight guards the division
    x = np.random.default_rng(1).uniform(size=(80, 3))
    y = np.zeros(80)
    model = train_classifier(
        x[:60], y[:60], x[60:], y[60:], ["a", "b", "c"], TrainConfig(max_epochs=5, patience=2)
    )
    probs = model.predict_proba(x)
    assert np.all((probs >= 0) & (probs <= 1))
