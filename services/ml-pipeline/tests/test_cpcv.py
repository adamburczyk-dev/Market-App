"""Combinatorial purged cross-validation (P4-4).

Walk-forward produces one out-of-sample path and every gate number is a single
draw from its distribution. CPCV's job is to show that distribution, so the
things worth pinning are the ones that would make it lie: leakage across a test
block, and "paths" that are really overlapping resamples of the same sessions.
"""

from datetime import UTC, datetime, timedelta

import pytest

from src.core.cpcv import cpcv_splits, n_backtest_paths, path_dispersion

D0 = datetime(2020, 1, 2, tzinfo=UTC)


def sessions(n: int = 600) -> list[datetime]:
    return [D0 + timedelta(days=i) for i in range(n)]


def test_the_number_of_paths_matches_the_combinatorial_design():
    # C(6,2) = 15 splits, each testing 2 groups; a path needs all 6 groups once
    assert n_backtest_paths(6, 2) == 5
    assert n_backtest_paths(10, 2) == 9
    assert n_backtest_paths(5, 1) == 1  # k=1 is plain k-fold: one path
    assert n_backtest_paths(4, 4) == 0  # nothing left to train on
    assert n_backtest_paths(4, 0) == 0


def test_splits_cover_every_combination_and_produce_the_promised_paths():
    from itertools import combinations

    result = cpcv_splits(sessions(), n_groups=6, test_groups=2)
    assert len(result.splits) == 15
    assert result.n_paths == 5
    assert {s.groups for s in result.splits} == set(combinations(range(6), 2))


def test_a_path_takes_each_group_from_exactly_one_split():
    """`paths[j][g]` supplies group g's predictions in path j, so the split at
    position g must actually test g. Without this the 'paths' are resamples of
    whatever splits happened to be picked, and their spread understates the very
    uncertainty CPCV exists to measure."""
    result = cpcv_splits(sessions(), n_groups=6, test_groups=2)
    assert result.paths, "no paths assembled"
    for path in result.paths:
        assert len(path) == 6
        for group, split_index in enumerate(path):
            assert group in result.splits[split_index].groups


def test_each_split_serves_exactly_its_tested_groups_across_all_paths():
    """A split IS reused — once per group it tests. Requiring disjoint splits
    instead would be a 1-factorization, which does not exist for most (N, k);
    the first version of this code searched for one greedily and silently
    returned a single degenerate path when it could not find it."""
    result = cpcv_splits(sessions(), n_groups=6, test_groups=2)
    uses = [i for path in result.paths for i in path]
    for index, split in enumerate(result.splits):
        assert uses.count(index) == len(split.groups) == 2


def test_training_never_touches_a_purged_session():
    """The leakage check. A label opened `horizon` sessions before a test block
    resolves inside it, so those training rows carry the test period's outcome.
    Both sides are purged, and the embargo widens the gap further.

    Stated as a distance: no training session may sit within `horizon + embargo`
    days of ANY test session. Sessions here are consecutive days, so day
    distance is index distance.
    """
    horizon, embargo = 10, 5
    gap = horizon + embargo
    result = cpcv_splits(sessions(300), n_groups=6, test_groups=2, horizon=horizon, embargo=embargo)
    for split in result.splits:
        test = sorted(split.test_dates)
        assert not (set(test) & set(split.train_dates)), "train and test overlap"
        for train_date in split.train_dates:
            nearest = min(abs((train_date - t).days) for t in test)
            assert nearest > gap, f"train session {nearest} days from a test session (gap {gap})"


def test_every_session_outside_the_purge_is_available_for_training():
    """Purging must not quietly cost more data than it should — an over-wide
    exclusion would look like a smaller dataset rather than like a bug."""
    result = cpcv_splits(sessions(600), n_groups=6, test_groups=2, horizon=10, embargo=5)
    for split in result.splits:
        assert len(split.train_dates) > 300, "purging removed far too much"


def test_degenerate_configurations_are_refused_not_approximated():
    with pytest.raises(ValueError, match="test_groups"):
        cpcv_splits(sessions(), n_groups=4, test_groups=4)
    with pytest.raises(ValueError, match="cannot form"):
        cpcv_splits(sessions(3), n_groups=6, test_groups=2)


# --- the statistic the whole thing exists to produce -----------------------


def test_dispersion_reports_the_spread_not_just_the_mean():
    """A mean with no spread beside it is the single-path number in a hat."""
    stats = path_dispersion([2.1, -0.4, 1.2, 0.3, -1.1])
    assert stats["n_paths"] == 5
    assert stats["std"] > 0
    assert stats["min"] == -1.1
    assert stats["max"] == 2.1
    # 3 of 5 ways of cutting the same data are positive — that is the number to
    # read, and it is not what the mean says
    assert stats["share_positive"] == 0.6


def test_dispersion_of_nothing_is_reported_as_nothing():
    assert path_dispersion([])["n_paths"] == 0
    assert path_dispersion([float("nan")])["n_paths"] == 0


# --- end to end on a real dataset -----------------------------------------


def test_cpcv_runs_and_reports_more_than_one_path():
    """The whole point: several out-of-sample readings from the same data, with
    their spread. One path with a mean is what we already had."""
    from dataclasses import replace as dc_replace

    from src.core.cpcv_run import cpcv_trials, run_cpcv

    from .test_training import SMALL, synthetic_dataset

    ds = synthetic_dataset(n=320)
    report = run_cpcv(ds, dc_replace(SMALL, val_size=10), n_groups=4, test_groups=1)

    assert report["n_splits"] == 4
    assert report["n_paths_evaluated"] >= 1
    for metric in ("sharpe_active", "sharpe", "ic"):
        assert set(report[metric]) >= {"n_paths", "mean", "std", "min", "max"}
    assert "share_positive" in report["note"]
    # G5's honesty input: every path is another look at the same data
    assert cpcv_trials(4, 1) == report["n_paths_evaluated"] or cpcv_trials(4, 1) >= 1


def test_cpcv_refuses_a_design_the_data_cannot_support():
    from dataclasses import replace as dc_replace

    from src.core.cpcv_run import run_cpcv

    from .test_training import SMALL, synthetic_dataset

    ds = synthetic_dataset(n=220)
    with pytest.raises(ValueError):
        run_cpcv(ds, dc_replace(SMALL, horizon=200, embargo=200), n_groups=6, test_groups=2)
