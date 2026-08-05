"""The excess label had to be BUILT, not switched on.

`LabelParams.excess` existed and was read by exactly one module — the target
study. `build_dataset` called `triple_barrier_label` unconditionally, so
setting the flag changed nothing about training: the excess label was a
measurement instrument wearing a pipeline's clothes.

These tests pin the two properties that make the benchmark honest, both of
which the length-matched predecessor got wrong on a real panel.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from trading_common.schemas import OHLCVBar

from src.core.dataset import DatasetParams, build_dataset
from src.core.labels import LabelParams
from src.core.market_path import MIN_CROSS_SECTION, market_levels, project_levels

START = datetime(2020, 1, 1, tzinfo=UTC)


def bars(symbol: str, closes: list[float], first_day: int = 0) -> list[OHLCVBar]:
    """A price path with no intraday range, so barriers can only be crossed at close."""
    return [
        OHLCVBar(
            symbol=symbol,
            timestamp=START + timedelta(days=first_day + i),
            open=c,
            high=c,
            low=c,
            close=c,
            volume=1_000_000,
            interval="1d",
        )
        for i, c in enumerate(closes)
    ]


def test_the_benchmark_is_the_median_of_daily_returns_not_of_cumulative_ratios():
    """A rebalanced index, so a long-history name cannot dominate the level.

    Three names each rise 1%/day; a fourth doubles on one day. The median
    return is unmoved by the outlier, which is the whole reason for a median.
    """
    steady = [100.0 * 1.01**i for i in range(10)]
    spike = [100.0] * 5 + [200.0] * 5
    panel = {
        "A": bars("A", steady),
        "B": bars("B", steady),
        "C": bars("C", steady),
        "D": bars("D", spike),
    }
    levels = market_levels(panel)
    sessions = sorted(levels)
    growth = levels[sessions[-1]] / levels[sessions[0]]
    assert growth == pytest.approx(1.01**9, rel=1e-6)


def test_a_late_lister_still_gets_a_benchmark_of_the_right_length():
    """The predecessor's failure mode, made explicit.

    It kept only series whose length equalled the longest, then guarded each
    label with `len(market) == n`. A name that listed halfway through therefore
    had NO usable benchmark and fell through to the absolute label — inside a
    result labelled "excess".
    """
    long_path = [100.0 + i for i in range(40)]
    panel = {
        "A": bars("A", long_path),
        "B": bars("B", long_path),
        "C": bars("C", long_path),
        "LATE": bars("LATE", [50.0 + i for i in range(20)], first_day=20),
    }
    levels = market_levels(panel)
    late_bars = panel["LATE"]
    projected = project_levels(levels, late_bars)

    assert len(projected) == len(late_bars)
    assert np.all(projected > 0)
    # And it is the SAME index the long names see over the overlapping dates,
    # not a private one rebased to the late lister's first bar.
    assert projected[0] == pytest.approx(levels[late_bars[0].timestamp])


def test_a_thin_cross_section_carries_the_level_instead_of_inventing_one():
    """With fewer than MIN_CROSS_SECTION names the median is a single stock.

    Calling that "the market" would make its own excess return zero by
    construction — a label that says nothing, indistinguishable from one that
    says the name matched the market.
    """
    panel = {"ONLY": bars("ONLY", [100.0, 150.0, 225.0])}
    levels = market_levels(panel)
    assert MIN_CROSS_SECTION > 1
    assert set(levels.values()) == {1.0}, "a one-name market must not move the benchmark"


def test_build_dataset_actually_uses_the_excess_label():
    """The test that fails on the code this commit replaces.

    A name that rises while the universe rises FASTER is a loser in excess
    terms and a winner in absolute terms. Under the old `build_dataset` — which
    called `triple_barrier_label` regardless of the flag — both label runs
    returned identical labels.
    """
    n = 90
    # The universe compounds at 1%/day; RUNNER manages only 0.2%/day.
    fast = [100.0 * 1.01**i for i in range(n)]
    slow = [100.0 * 1.002**i for i in range(n)]
    panel = {
        "FAST1": bars("FAST1", fast),
        "FAST2": bars("FAST2", fast),
        "FAST3": bars("FAST3", fast),
        "RUNNER": bars("RUNNER", slow),
    }

    # BOTH sides explicit. Pinning only the horizon was not enough: once
    # `excess` became the default, `LabelParams(horizon=10)` silently produced
    # an EXCESS dataset and this test compared excess against excess. A test
    # about whether a flag changes behaviour cannot lean on that flag's default.
    common = {"min_history": 30, "min_universe": 3}
    absolute = build_dataset(
        panel, DatasetParams(label=LabelParams(horizon=10, excess=False), **common)
    )
    excess = build_dataset(
        panel, DatasetParams(label=LabelParams(horizon=10, excess=True), **common)
    )

    def labels_for(ds, symbol: str) -> list[float]:
        return [y for y, s in zip(ds.y, ds.symbols, strict=True) if s == symbol]

    runner_absolute = labels_for(absolute, "RUNNER")
    runner_excess = labels_for(excess, "RUNNER")
    assert runner_absolute and runner_excess, "fixture produced no RUNNER rows"
    # Rising on its own: every absolute label is a win.
    assert set(runner_absolute) == {1.0}
    # Losing to the cross-section: every excess label is a loss.
    assert set(runner_excess) == {0.0}


def test_next_returns_stay_absolute_under_an_excess_label():
    """The book's P&L is money, and money is absolute.

    `relative_metrics` already subtracts the equal-weight universe to produce
    sharpe_active. Making next_returns excess as well would subtract the
    benchmark twice and quietly change what the gate's economics condition
    reads.
    """
    n = 90
    fast = [100.0 * 1.01**i for i in range(n)]
    slow = [100.0 * 1.002**i for i in range(n)]
    panel = {
        "FAST1": bars("FAST1", fast),
        "FAST2": bars("FAST2", fast),
        "FAST3": bars("FAST3", fast),
        "RUNNER": bars("RUNNER", slow),
    }
    ds = build_dataset(
        panel,
        DatasetParams(label=LabelParams(horizon=10, excess=True), min_history=30, min_universe=3),
    )
    runner = [r for r, s in zip(ds.next_returns, ds.symbols, strict=True) if s == "RUNNER"]
    assert runner
    # RUNNER's own price rises 0.2%/day. Excess returns would be NEGATIVE here.
    assert all(r > 0 for r in runner)
    assert np.mean(runner) == pytest.approx(0.002, rel=1e-3)
