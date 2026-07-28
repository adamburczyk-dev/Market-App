"""Overlapping labels are one market episode, not `horizon` episodes."""

import numpy as np
import pytest

from src.core.dataset import build_dataset
from src.core.model import TrainConfig, train_classifier
from src.core.uniqueness import average_uniqueness, effective_rows

from .test_dataset import PARAMS, universe


def test_isolated_labels_keep_full_weight():
    spans = [("A", 0, 0), ("A", 5, 5), ("A", 10, 10)]
    assert np.allclose(average_uniqueness(spans), 1.0)


def test_fully_overlapping_labels_share_one_episode():
    # Ten labels all spanning the same ten sessions: each is worth a tenth.
    spans = [("A", i, i + 9) for i in range(10)]
    weights = average_uniqueness(spans)
    assert weights.min() > 0
    assert effective_rows(weights) == pytest.approx(1.9, abs=0.3)  # far below 10
    assert weights[0] > weights[4]  # the edges overlap fewer neighbours


def test_symbols_are_counted_separately():
    # Two symbols labeled on the same dates are two observations, not one.
    per_symbol = average_uniqueness([("A", 0, 9)])
    together = average_uniqueness([("A", 0, 9), ("B", 0, 9)])
    assert np.allclose(together, per_symbol[0])


def test_dataset_weights_reflect_the_horizon():
    ds = build_dataset(universe(), PARAMS)
    assert len(ds.weights) == ds.n_samples
    assert ds.weights.max() <= 1.0 + 1e-9
    # h=10 daily sampling → the weighted sample is a fraction of the row count
    assert effective_rows(ds.weights) < ds.n_samples / 3


def test_training_accepts_and_validates_weights():
    rng = np.random.default_rng(0)
    x = rng.random((200, 4))
    y = (x[:, 0] > 0.5).astype(float)
    cfg = TrainConfig(hidden=(8, 4), max_epochs=10, min_epochs=5, patience=5)

    model = train_classifier(x, y, x, y, ["a", "b", "c", "d"], cfg, sample_weights=np.ones(200))
    assert model.predict_proba(x).shape == (200,)

    with pytest.raises(ValueError, match="sample_weights"):
        train_classifier(x, y, x, y, ["a", "b", "c", "d"], cfg, sample_weights=np.ones(5))


def test_weights_change_what_the_model_learns():
    """Down-weighting a group must actually move the fit, otherwise the whole
    mechanism is decoration."""
    rng = np.random.default_rng(3)
    x = np.vstack([rng.random((150, 2)), rng.random((150, 2))])
    y = np.concatenate([np.ones(150), np.zeros(150)])  # group A up, group B down
    x[:150, 0] += 0.5  # separable
    cfg = TrainConfig(hidden=(8, 4), max_epochs=40, min_epochs=20, patience=20, seed=1)

    weights = np.concatenate([np.full(150, 0.01), np.ones(150)])  # A almost ignored
    flat = train_classifier(x, y, x, y, ["a", "b"], cfg).predict_proba(x[:150]).mean()
    tilted = (
        train_classifier(x, y, x, y, ["a", "b"], cfg, sample_weights=weights)
        .predict_proba(x[:150])
        .mean()
    )
    assert tilted < flat, "down-weighted rows still dominated the fit"
