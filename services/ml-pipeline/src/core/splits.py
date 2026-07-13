"""Purged walk-forward splits for overlapping labels (López de Prado).

Triple-barrier labels span up to ``horizon`` sessions, so a training sample
whose label window reaches into the test period leaks future information
through the target. Purging drops the last ``horizon`` training sessions
before each test block; the embargo widens that gap further to damp serial
correlation between the last usable training features and the first test
features. Splits are strictly forward (train precedes test) — never random,
per the project rule.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold[T]:
    train_dates: tuple[T, ...]  # ordered session keys (dates)
    test_dates: tuple[T, ...]


def purged_walk_forward[T](
    dates: Sequence[T],
    train_size: int,
    test_size: int,
    horizon: int,
    embargo: int = 0,
    step: int | None = None,
) -> list[Fold[T]]:
    """Forward folds over sorted unique session dates.

    Each fold's test block is ``test_size`` sessions; its training window is
    the ``train_size`` sessions immediately before it MINUS the trailing
    ``horizon + embargo`` sessions (purge + embargo). ``step`` defaults to
    ``test_size`` (non-overlapping test blocks). Folds whose purged training
    window would be empty are not produced.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    gap = horizon + embargo
    if gap >= train_size:
        raise ValueError("purge+embargo consume the whole training window")

    ordered = list(dates)
    step = step or test_size
    folds: list[Fold[T]] = []
    test_start = train_size
    while test_start + test_size <= len(ordered):
        train = ordered[test_start - train_size : test_start - gap]
        test = ordered[test_start : test_start + test_size]
        folds.append(Fold(train_dates=tuple(train), test_dates=tuple(test)))
        test_start += step
    return folds
