"""The capacity probe must give opposite verdicts on the two cases that matter."""

from datetime import UTC, datetime, timedelta

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


def make_dataset(x: np.ndarray, y: np.ndarray, names_per_session: int = 40) -> Dataset:
    """Rows carry a session, because the probe's control shuffles WITHIN one."""
    n = len(y)
    sessions = [
        datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i // names_per_session) for i in range(n)
    ]
    return Dataset(
        x=x,
        y=y,
        next_returns=np.zeros(n),
        dates=sessions,
        symbols=[f"S{i % names_per_session:03d}" for i in range(n)],
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
    assert probe.separated, "real and control fits must not overlap for a positive verdict"
    assert "EXISTS" in probe.verdict


def test_probe_repeats_both_sides():
    """One fit per side cannot separate a small gap from seed noise — the first
    real run landed at +0.0518 against a 0.05 threshold."""
    probe = run_capacity_probe(
        learnable(), production_config=PRODUCTION, overfit_config=PROBE, n_real=2, n_shuffles=3
    )
    assert len(probe.real_runs) == 2
    assert len(probe.shuffled_runs) == 3
    assert len(set(probe.shuffled_runs)) == 3, "each control must be a fresh permutation"
    assert probe.auc_train_real == pytest.approx(sum(probe.real_runs) / 2)
    assert probe.margin == pytest.approx(min(probe.real_runs) - max(probe.shuffled_runs))


def test_overlapping_runs_are_called_inconclusive():
    """A mean gap over the threshold whose runs overlap is not a verdict."""
    import src.core.capacity as capacity

    # real .62/.54 (mean .58), controls .55/.50/.51 (mean .52) -> gap .06, but the
    # worst real fit (.54) sits BELOW the best control (.55): the runs overlap.
    scores = iter([0.62, 0.54, 0.55, 0.50, 0.51, 0.53])
    original = capacity._fit_and_score
    capacity._fit_and_score = lambda *a, **k: next(scores)  # type: ignore[assignment]
    try:
        probe = run_capacity_probe(learnable(), n_real=2, n_shuffles=3)
    finally:
        capacity._fit_and_score = original  # type: ignore[assignment]

    assert probe.gap >= 0.05
    assert not probe.separated
    assert "INCONCLUSIVE" in probe.verdict


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


def date_effect_only(n_sessions: int = 120, n_names: int = 60, seed: int = 3) -> Dataset:
    """No cross-sectional predictability whatsoever, but strong DATE effects.

    Features are persistent and partly load on a market factor; the label is
    that market factor, identical in distribution for every name on a session.
    Ranking names by any feature therefore says nothing about which of them
    goes up — the only thing a model can learn is which date a row belongs to.
    """
    rng = np.random.default_rng(seed)
    market = np.zeros(n_sessions)
    for t in range(1, n_sessions):
        market[t] = 0.97 * market[t - 1] + rng.normal(0, 0.3)

    state = rng.normal(0, 1, (n_names, 6))
    x = np.zeros((n_sessions * n_names, 6))
    y = np.zeros(n_sessions * n_names)
    row = 0
    for t in range(n_sessions):
        state = 0.95 * state + rng.normal(0, 0.3, state.shape)
        loaded = state + market[t] * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        x[row : row + n_names] = loaded.argsort(axis=0).argsort(axis=0) / (n_names - 1)
        p_up = 1.0 / (1.0 + np.exp(-2.0 * market[t]))
        y[row : row + n_names] = (rng.random(n_names) < p_up).astype(float)
        row += n_names
    return make_dataset(x, y, names_per_session=n_names)


def test_a_pure_date_effect_is_not_reported_as_structure():
    """The control must permute WITHIN a session, not globally.

    A triple-barrier label is largely a date effect (pairwise correlation 0.36
    on the real panel) and features are persistent, so a big model can learn
    "this configuration is that date, and that date rose" with no
    cross-sectional edge at all. A global shuffle destroys the date effect too,
    so the control cannot do that — and the gap looks like signal. Measured on
    this fixture before the fix: +0.066 against a 0.05 threshold, separated,
    verdict "learnable structure EXISTS", on data that has none by
    construction.
    """
    probe = run_capacity_probe(
        date_effect_only(), production_config=PRODUCTION, overfit_config=PROBE
    )
    assert not (probe.gap >= 0.05 and probe.separated), (
        f"date effects alone were reported as structure: gap {probe.gap:+.4f}"
    )
    assert "NO learnable structure" in probe.verdict


def test_a_control_that_cannot_find_its_sessions_is_an_error():
    """Silently permuting nothing would make every probe answer "no structure"
    — wrong, and quiet about it."""
    ds = learnable()
    broken = Dataset(
        x=ds.x,
        y=ds.y,
        next_returns=ds.next_returns,
        dates=[],
        symbols=ds.symbols,
        feature_names=ds.feature_names,
    )
    with pytest.raises(ValueError, match="dates has 0 rows"):
        run_capacity_probe(broken, production_config=PRODUCTION, overfit_config=PROBE)
