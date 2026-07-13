"""Tests for purged walk-forward splits."""

import pytest

from src.core.splits import purged_walk_forward

DATES = list(range(100))  # session keys — any ordered hashables


def test_forward_non_overlapping_folds():
    folds = purged_walk_forward(DATES, train_size=40, test_size=20, horizon=10)
    assert len(folds) == 3  # tests at [40,60), [60,80), [80,100)
    for fold in folds:
        assert max(fold.train_dates) < min(fold.test_dates)
    assert folds[0].test_dates[0] == 40
    assert folds[1].test_dates[0] == 60


def test_purge_removes_label_overlap():
    folds = purged_walk_forward(DATES, train_size=40, test_size=20, horizon=10)
    first = folds[0]
    # last training date must be at least `horizon` sessions before the test start
    assert first.test_dates[0] - max(first.train_dates) >= 10
    assert len(first.train_dates) == 30  # 40 − purge 10


def test_embargo_widens_the_gap():
    folds = purged_walk_forward(DATES, train_size=40, test_size=20, horizon=10, embargo=5)
    first = folds[0]
    assert first.test_dates[0] - max(first.train_dates) >= 15
    assert len(first.train_dates) == 25


def test_custom_step_slides_tests():
    folds = purged_walk_forward(DATES, train_size=40, test_size=20, horizon=5, step=10)
    starts = [fold.test_dates[0] for fold in folds]
    assert starts == [40, 50, 60, 70, 80]


def test_too_short_history_yields_no_folds():
    assert purged_walk_forward(list(range(50)), train_size=40, test_size=20, horizon=5) == []


def test_degenerate_gap_raises():
    with pytest.raises(ValueError, match="purge"):
        purged_walk_forward(DATES, train_size=10, test_size=5, horizon=8, embargo=3)


def test_invalid_sizes_raise():
    with pytest.raises(ValueError):
        purged_walk_forward(DATES, train_size=0, test_size=5, horizon=1)
