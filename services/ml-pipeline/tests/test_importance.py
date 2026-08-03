"""Permutation importance: what the model uses, and the two ways to get it wrong.

The instrument is checked against models whose behaviour is known exactly, so a
failure points at the measurement rather than at a fit that had a bad seed.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from src.core.evaluation import session_groups
from src.core.importance import (
    NOISE_FEATURE,
    feature_groups,
    noise_control_column,
    permutation_importance,
    sidak_tstat_bar,
)

SESSIONS = 60
NAMES = 20


@dataclass
class FakeModel:
    """A model whose dependence on each column is known by construction."""

    feature_names: list[str]
    score: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.score(x), dtype=float)


def panel(
    n_sessions: int = SESSIONS, n_names: int = NAMES, seed: int = 3
) -> tuple[np.ndarray, np.ndarray, list[datetime], np.ndarray]:
    """(x, y, dates, next_returns) for a universe with one real signal.

    Column 0 (`signal`) ranks the forward return within every session; column 1
    (`noise`) is an independent draw; column 2 (`regime`) is CONSTANT inside a
    session — the shape every `macro_*` one-hot has.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2, tzinfo=UTC)
    dates: list[datetime] = []
    rows: list[list[float]] = []
    returns: list[float] = []
    for s in range(n_sessions):
        session = start + timedelta(days=s)
        signal = rng.permutation((np.arange(n_names) + 0.5) / n_names)
        regime = 1.0 if s % 2 == 0 else 0.0
        for i in range(n_names):
            dates.append(session)
            rows.append([float(signal[i]), float(rng.random()), regime])
            # the forward return is a monotone function of the signal plus noise
            returns.append(float(signal[i]) - 0.5 + 0.05 * rng.standard_normal())
    x = np.asarray(rows, dtype=float)
    next_returns = np.asarray(returns, dtype=float)
    y = (next_returns > 0).astype(float)
    return x, y, dates, next_returns


NAMES_3 = ["signal", "noise", "regime"]


def measure(score, n_repeats: int = 4, seed: int = 5, names: list[str] | None = None, **kwargs):
    x, y, dates, returns = panel(seed=seed)
    feature_names = names or NAMES_3
    return permutation_importance(
        FakeModel(feature_names, score),
        x,
        y,
        dates,
        returns,
        feature_names,
        n_repeats=n_repeats,
        **kwargs,
    )


def entry(report, name: str):
    return next(e for e in report.features if e.name == name)


def test_used_column_drops_the_ic_and_ignored_columns_do_not():
    report = measure(lambda x: x[:, 0])
    used, ignored = entry(report, "signal"), entry(report, "noise")
    assert report.base_ic > 0.9  # the fake model IS the signal
    assert used.ic_drop > 0.5
    assert used.tstat > report.tstat_bar
    # Permuting a column the model never reads cannot change one prediction.
    assert ignored.ic_drop == pytest.approx(0.0, abs=1e-12)
    assert ignored.tstat == 0.0


def test_a_session_constant_feature_scores_zero_and_a_global_shuffle_would_not():
    """The lesson from the capacity probe, in the shape it takes here.

    `regime` is constant inside a session, so it cannot possibly order a
    cross-section — and the model below uses it as a multiplier, meaning a
    permutation that MOVED values between sessions would hand some rows 0 and
    others 1 within one day and register a large, entirely spurious drop. Every
    `macro_*` one-hot has exactly this shape.
    """
    report = measure(lambda x: x[:, 0] * x[:, 2])
    assert entry(report, "regime").ic_drop == pytest.approx(0.0, abs=1e-12)

    # What a global permutation would have reported, measured rather than asserted.
    x, y, dates, returns = panel(seed=5)
    rng = np.random.default_rng(0)
    scrambled = x.copy()
    scrambled[:, 2] = rng.permutation(x[:, 2])
    groups = session_groups(dates)
    from src.core.evaluation import session_ic_series

    base = session_ic_series(x[:, 0] * x[:, 2], returns, groups)
    globally = session_ic_series(scrambled[:, 0] * scrambled[:, 2], returns, groups)
    assert float((base - globally).mean()) > 0.2


def test_duplicated_columns_split_the_credit_and_the_family_row_recovers_it():
    """Two identical inputs each look weak; permuted together they do not."""
    names = ["a", "a_copy", "unused"]
    score = lambda x: (x[:, 0] + x[:, 1]) / 2.0  # noqa: E731
    x, y, dates, returns = panel(seed=7)
    x[:, 1] = x[:, 0]  # a perfect duplicate
    model = FakeModel(names, score)
    report = permutation_importance(
        model,
        x,
        y,
        dates,
        returns,
        names,
        n_repeats=4,
        groups={"duplicated": ("a", "a_copy")},
    )
    single = entry(report, "a")
    family = next(e for e in report.groups if e.name == "duplicated")
    assert family.ic_drop > single.ic_drop
    # And the table says WHY the single number is small.
    assert single.max_abs_correlation == pytest.approx(1.0, abs=1e-9)
    assert single.most_correlated_with == "a_copy"
    assert single.redundant
    assert "Credit is split" in report.verdict


def test_the_bar_tightens_with_the_number_of_tests():
    assert sidak_tstat_bar(1) == pytest.approx(1.959964, abs=1e-5)
    assert sidak_tstat_bar(15) > sidak_tstat_bar(5) > sidak_tstat_bar(1)
    # A single feature's report must not be judged against a 15-test bar.
    report = measure(lambda x: x[:, 0])
    assert report.tstat_bar == pytest.approx(sidak_tstat_bar(3), abs=1e-9)


def test_a_holdout_too_short_to_pair_refuses_instead_of_reporting_ratios():
    x, y, dates, returns = panel(n_sessions=5)
    with pytest.raises(ValueError, match="rankable sessions"):
        permutation_importance(FakeModel(NAMES_3, lambda z: z[:, 0]), x, y, dates, returns, NAMES_3)


def test_noise_control_column_is_a_valid_rank_and_carries_nothing():
    _, _, dates, _ = panel()
    column = noise_control_column(dates, seed=2)
    for rows in session_groups(dates):
        values = np.sort(column[rows])
        expected = (np.arange(len(rows)) + 0.5) / len(rows)
        # exactly the construction of a cross-sectional rank, not merely its range
        assert np.allclose(values, expected)

    names = [*NAMES_3, NOISE_FEATURE]
    x, y, dts, returns = panel()
    x = np.column_stack([x, column])
    report = permutation_importance(
        FakeModel(names, lambda z: z[:, 0]), x, y, dts, returns, names, n_repeats=3
    )
    assert report.noise_control is not None
    assert abs(report.noise_control.tstat) < report.tstat_bar
    # And nothing the model ignores may be reported as significant either — the
    # repeats are summed and divided, so identical predictions leave dust behind
    # and dust/dust once produced t = +3.23 for a column never read.
    assert all(e.tstat == 0.0 for e in report.features if e.name != "signal")
    # The planted column is a measurement, not a test the bar has to widen for.
    assert report.tstat_bar == pytest.approx(sidak_tstat_bar(3), abs=1e-9)
    assert "noise" in report.verdict


def test_groups_are_derived_from_what_the_columns_measure():
    groups = feature_groups(
        ["macro_crisis", "macro_expansion", "return_1d", "return_5d", "rsi_14", "orphan"]
    )
    assert groups["macro_regime"] == ("macro_crisis", "macro_expansion")
    assert groups["momentum"] == ("return_1d", "return_5d")
    # One member is not a family: it would duplicate that column's own row.
    assert "oscillators" not in groups
    assert "orphan" not in {m for members in groups.values() for m in members}


def test_session_groups_are_in_date_order():
    """The Newey-West correction and the paired difference both assume it."""
    start = datetime(2024, 5, 1, tzinfo=UTC)
    later, earlier = start + timedelta(days=1), start
    dates = [later] * 3 + [earlier] * 3
    assert session_groups(dates) == [[3, 4, 5], [0, 1, 2]]
