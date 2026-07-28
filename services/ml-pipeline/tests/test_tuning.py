"""The sweep must select on folds only and carry its trial count forward."""

import pytest

from src.core.dataset import DatasetParams, build_dataset, drop_zero_variance_features
from src.core.labels import LabelParams
from src.core.model import TrainConfig
from src.core.training import TrainingParams
from src.core.tuning import DEFAULT_GRID, run_sweep

from .test_training import WIDE, interaction_universe

GRID = {
    "small": TrainConfig(hidden=(16, 8), max_epochs=40, min_epochs=15, patience=8),
    "wider": TrainConfig(hidden=(64, 32), dropout=0.1, max_epochs=40, min_epochs=15, patience=8),
}


@pytest.fixture(scope="module")
def dataset():
    ds = build_dataset(
        interaction_universe(n_sessions=640),
        DatasetParams(label=LabelParams(), min_history=60, min_universe=20),
    )
    cleaned, _ = drop_zero_variance_features(ds)
    return cleaned


@pytest.mark.slow
def test_sweep_ranks_candidates_and_reports_trials(dataset):
    report = run_sweep(dataset, WIDE, grid=GRID, n_folds=2)
    assert [c.name for c in report.candidates] == sorted(
        [c.name for c in report.candidates],
        key=lambda n: -next(c.ic_tstat for c in report.candidates if c.name == n),
    )
    assert report.n_trials == len(GRID), "the gate must be told how many configs were tried"
    assert all(c.n_folds > 0 for c in report.candidates)
    # On a universe with a real edge the winner has a positive IC t-stat.
    assert report.best is not None
    assert report.candidates[0].ic_tstat > 0


@pytest.mark.slow
def test_sweep_never_touches_the_holdout(dataset, monkeypatch):
    """Selection on the holdout would make the final gate meaningless. Pin it by
    recording every session the sweep ever scores."""
    import src.core.tuning as tuning

    seen: set = set()
    original = tuning.score_window

    def spy(ds, model, test_dates, name, n_train, params, fit_dates=None):  # type: ignore[no-untyped-def]
        seen.update(test_dates)
        if fit_dates:
            seen.update(fit_dates)
        return original(ds, model, test_dates, name, n_train, params, fit_dates=fit_dates)

    monkeypatch.setattr(tuning, "score_window", spy)
    params = WIDE
    run_sweep(dataset, params, grid={"small": GRID["small"]}, n_folds=2)

    holdout = set(sorted(set(dataset.dates))[-params.holdout_size :])
    assert seen, "precondition: the sweep scored something"
    assert not (seen & holdout), "the sweep saw holdout sessions"


def test_sweep_refuses_a_dataset_too_short_to_split(dataset):
    with pytest.raises(ValueError, match="sessions"):
        run_sweep(dataset, TrainingParams())  # production window sizes need ~945 sessions


def test_default_grid_spans_the_axes_run_2_implicates():
    # Capacity and regularization are the two things a flat train AUC points at;
    # a grid that varies neither cannot answer the question it exists for.
    hidden = {c.hidden for c in DEFAULT_GRID.values()}
    dropout = {c.dropout for c in DEFAULT_GRID.values()}
    assert len(hidden) > 1
    assert len(dropout) > 1
    assert "production" in DEFAULT_GRID, "the incumbent must be in the comparison"
