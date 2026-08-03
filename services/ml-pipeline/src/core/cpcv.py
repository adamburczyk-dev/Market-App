"""Combinatorial purged cross-validation (P4-4, AFML ch. 12).

Walk-forward gives ONE out-of-sample path. Every number the gate reads —
Sharpe, IC, the deflated Sharpe over the stitched curve — is a single draw from
that path's distribution, and run #2 showed how wide that distribution is: fold
Sharpes from −1.61 to +4.54 on the same data and the same configuration.

CPCV splits the timeline into N groups and tests on every combination of k of
them, purging and embargoing around each test block. With N=6, k=2 that is 15
splits which recombine into 5 distinct out-of-sample PATHS rather than one. The
spread across those paths is the answer to "how much of this number is the
particular way I cut the data", and it is the only honest input to a statement
like "the strategy has a Sharpe of X".

Two properties this must have, both easy to get wrong:

- **Purging is per test block, not per split.** Each of the k test groups needs
  its own gap on both sides; a label started `horizon` sessions before a test
  group ends inside it and leaks.
- **A path takes each group's predictions from exactly one split.** Every group
  appears in the same number of splits, so the j-th of them forms path j. A
  split serves several paths — once per group it tests — and trying instead to
  build paths from mutually disjoint splits is a 1-factorization that does not
  exist for most (N, k).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import structlog

from src.core.labels import LABEL_HORIZON

logger = structlog.get_logger()


@dataclass(frozen=True)
class CpcvSplit[T]:
    """One (train, test) split. `groups` names which groups form the test set."""

    train_dates: tuple[T, ...]
    test_dates: tuple[T, ...]
    groups: tuple[int, ...]


@dataclass(frozen=True)
class CpcvPaths[T]:
    splits: list[CpcvSplit[T]] = field(default_factory=list)
    # paths[j][g] = index of the split supplying group g's predictions in path j
    paths: list[tuple[int, ...]] = field(default_factory=list)
    n_groups: int = 0
    test_groups: int = 0

    @property
    def n_paths(self) -> int:
        return len(self.paths)


def n_backtest_paths(n_groups: int, test_groups: int) -> int:
    """How many distinct OOS paths N groups taken k at a time produce.

    C(N,k) splits each contribute k tested groups, and a path needs each of the
    N groups exactly once: C(N,k) * k / N.
    """
    if not 0 < test_groups < n_groups:
        return 0
    return math.comb(n_groups, test_groups) * test_groups // n_groups


def cpcv_splits[T](
    dates: Sequence[T],
    n_groups: int = 6,
    test_groups: int = 2,
    horizon: int = LABEL_HORIZON,
    embargo: int = 5,
) -> CpcvPaths[T]:
    """Every combination of `test_groups` groups as the test set, purged.

    Groups are contiguous blocks of the sorted session list, so training and
    test never interleave within a block and the purge has a well-defined side.
    """
    if not 0 < test_groups < n_groups:
        raise ValueError("test_groups must be in (0, n_groups)")
    ordered = list(dates)
    if len(ordered) < n_groups:
        raise ValueError(f"{len(ordered)} sessions cannot form {n_groups} groups")

    bounds = np.linspace(0, len(ordered), n_groups + 1).astype(int)
    blocks = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_groups)]
    gap = horizon + embargo

    splits: list[CpcvSplit[T]] = []
    for combo in combinations(range(n_groups), test_groups):
        test_index: list[int] = []
        excluded: set[int] = set()
        for g in combo:
            start, end = blocks[g]
            test_index.extend(range(start, end))
            # purge + embargo on BOTH sides of every test block: a label opened
            # before the block closes inside it, and features just after it are
            # still serially tied to the block's last sessions.
            excluded.update(range(max(0, start - gap), min(len(ordered), end + gap)))
        train_index = [i for i in range(len(ordered)) if i not in excluded]
        if not train_index or not test_index:
            continue
        splits.append(
            CpcvSplit(
                train_dates=tuple(ordered[i] for i in train_index),
                test_dates=tuple(ordered[i] for i in test_index),
                groups=combo,
            )
        )

    paths = _assemble_paths(splits, n_groups, test_groups)
    logger.info(
        "CPCV splits built",
        sessions=len(ordered),
        n_groups=n_groups,
        test_groups=test_groups,
        splits=len(splits),
        paths=len(paths),
    )
    return CpcvPaths(splits=splits, paths=paths, n_groups=n_groups, test_groups=test_groups)


def _assemble_paths(
    splits: Sequence[CpcvSplit[Any]], n_groups: int, test_groups: int
) -> list[tuple[int, ...]]:
    """Assign, for every path, which split supplies each group's predictions.

    `paths[j][g]` is the index of the split that provides group g's
    out-of-sample predictions in path j. Every group appears in exactly
    C(N-1, k-1) splits, which equals the path count, so taking the j-th split
    that tests g gives each path exactly one prediction per group and uses every
    split's every tested group exactly once across all paths.

    A split therefore SERVES SEVERAL PATHS — once for each group it tests. An
    earlier version tried to build paths from mutually disjoint splits instead,
    which is a 1-factorization: it does not exist for most (N, k), and greedy
    search silently returned one degenerate path where it failed.
    """
    by_group: dict[int, list[int]] = {g: [] for g in range(n_groups)}
    for i, split in enumerate(splits):
        for g in split.groups:
            by_group[g].append(i)

    n_paths = n_backtest_paths(n_groups, test_groups)
    paths: list[tuple[int, ...]] = []
    for j in range(n_paths):
        if any(len(by_group[g]) <= j for g in range(n_groups)):
            break  # a split was dropped (degenerate window) — stop rather than guess
        paths.append(tuple(by_group[g][j] for g in range(n_groups)))
    return paths


def path_dispersion(path_metrics: Sequence[float]) -> dict[str, float]:
    """What the spread across paths says — the reason CPCV exists.

    A mean with no spread beside it is the single-path number wearing a
    different hat. `share_positive` matters more than the mean: a strategy whose
    Sharpe is positive on 3 of 5 ways of cutting the same data has not been
    shown to work, whatever its average says.
    """
    values = np.asarray([v for v in path_metrics if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return {"n_paths": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n_paths": int(len(values)),
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "share_positive": round(float(np.mean(values > 0)), 4),
    }


__all__ = [
    "CpcvPaths",
    "CpcvSplit",
    "cpcv_splits",
    "n_backtest_paths",
    "path_dispersion",
]
