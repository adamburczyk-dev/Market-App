"""Point-in-time universe selection and the survivorship report (P3-1/P3-3).

Two things have to hold. The selection must decide membership from data
available BEFORE the rebalance — otherwise it is just another way to pick
winners. And the report must say so when the candidate list contains no
delistings, because then the mechanism is running on a survivor list and no
amount of correct code changes that.
"""

from datetime import UTC, date, datetime, timedelta

import numpy as np
from trading_common.features import FULL_HISTORY
from trading_common.schemas import Interval, OHLCVBar

from src.core.dataset import DatasetParams, build_dataset
from src.core.universe import (
    UniverseParams,
    build_universe,
    survivorship_report,
)

START = datetime(2020, 1, 2, tzinfo=UTC)
N = 400


def bars(
    symbol: str,
    volume: float,
    n: int = N,
    first_index: int = 0,
    seed: int = 1,
) -> list[OHLCVBar]:
    """`first_index` shifts the listing date; the series stops at n regardless."""
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, n))
    return [
        OHLCVBar(
            symbol=symbol,
            timestamp=START + timedelta(days=first_index + i),
            interval=Interval.D1,
            open=float(c),
            high=float(c) * 1.01,
            low=float(c) * 0.99,
            close=float(c),
            adj_close=float(c),
            volume=volume,
            source="test",
        )
        for i, c in enumerate(closes)
    ]


def liquid_universe(n_symbols: int = 30) -> dict[str, list[OHLCVBar]]:
    """Volume descends with the index, so the top-N is exactly predictable.

    One shared price path (same seed) on purpose: selection ranks on DOLLAR
    volume, so with per-symbol price paths the close would reorder names whose
    share volumes are close together and the expected top-N would stop being
    exactly predictable.
    """
    return {
        f"S{k:02d}": bars(f"S{k:02d}", volume=1_000_000.0 * (n_symbols - k), seed=1)
        for k in range(n_symbols)
    }


def test_top_n_is_chosen_by_liquidity_and_is_deterministic():
    u = build_universe(liquid_universe(), UniverseParams(top_n=10, min_history=60))
    populated = [m for m in u.members_by_date.values() if m]
    assert populated, "no rebalance produced a universe"
    for members in populated:
        assert len(members) == 10
        assert set(members) == {f"S{k:02d}" for k in range(10)}  # the ten busiest
    again = build_universe(liquid_universe(), UniverseParams(top_n=10, min_history=60))
    assert u.members_by_date == again.members_by_date


def test_early_rebalances_are_empty_until_the_history_requirement_is_met():
    """Not a defect: on the first rebalance nobody has a year of history yet, so
    nobody is eligible. Filling the gap with 'everyone' would put names into the
    cross-section with their slow features neutral-filled."""
    u = build_universe(liquid_universe(12), UniverseParams(top_n=5, min_history=FULL_HISTORY))
    dates = sorted(u.members_by_date)
    assert u.members_by_date[dates[0]] == ()
    assert len(u.members_by_date[dates[-1]]) == 5


def test_membership_follows_liquidity_over_time():
    """A name whose turnover collapses must leave at the next rebalance — and
    not before, since membership is held between rebalances."""
    universe = liquid_universe(24)
    fading = universe["S00"]  # the busiest name
    switch = 200
    universe["S00"] = [
        b.model_copy(update={"volume": 1.0}) if i >= switch else b for i, b in enumerate(fading)
    ]
    u = build_universe(universe, UniverseParams(top_n=5, rebalance_days=50, min_history=60))

    early = u.members_on(date_at(150))
    late = u.members_on(date_at(390))
    assert "S00" in early
    assert "S00" not in late


def date_at(index: int) -> date:
    return (START + timedelta(days=index)).date()


def test_before_the_first_rebalance_the_universe_is_empty_not_everyone():
    """Defaulting to 'all symbols' would quietly restore the survivor list on
    exactly the sessions the selection has not spoken for."""
    u = build_universe(liquid_universe(), UniverseParams(top_n=5, min_history=60))
    populated = sorted(d for d, m in u.members_by_date.items() if m)
    first = populated[0]
    assert u.members_on(first - timedelta(days=1)) == frozenset()
    assert len(u.members_on(first)) == 5


def test_selection_uses_only_history_up_to_the_rebalance():
    """The membership decision must not see the future. A name that is dead
    quiet until late and then explodes in volume cannot be in the early
    universe — if it is, the selection read data it should not have."""
    universe = liquid_universe(20)
    sleeper = "S19"  # lowest volume by construction
    switch = 300
    universe[sleeper] = [
        b.model_copy(update={"volume": 500_000_000.0}) if i >= switch else b
        for i, b in enumerate(universe[sleeper])
    ]
    u = build_universe(universe, UniverseParams(top_n=3, rebalance_days=50, min_history=60))
    assert sleeper not in u.members_on(date_at(100))
    assert sleeper in u.members_on(date_at(390))


def test_names_without_enough_history_cannot_join():
    """A short-history name would enter the cross-section with its slow features
    neutral-filled, which is not a measurement."""
    universe = liquid_universe(10)
    universe["NEW"] = bars("NEW", volume=10_000_000_000.0, n=100, first_index=300, seed=99)
    u = build_universe(universe, UniverseParams(top_n=5, min_history=FULL_HISTORY))
    assert "NEW" not in u.all_members
    assert "NEW" in u.diagnostics["never_selected"]


def test_diagnostics_describe_the_universe_that_was_built():
    u = build_universe(liquid_universe(30), UniverseParams(top_n=10, min_history=60))
    d = u.diagnostics
    assert d["candidates"] == 30
    assert d["top_n"] == 10
    assert d["universe_size_max"] == 10
    assert d["universe_size_min"] == 0  # the first rebalance predates min_history
    assert d["turnover_mean"] == 0.0  # a static ranking has no churn once populated
    assert len(d["never_selected"]) == 20


def test_empty_input_does_not_explode():
    u = build_universe({}, UniverseParams())
    assert u.members_by_date == {}
    assert u.members_on(date(2024, 1, 1)) == frozenset()


# --- the dataset actually respects membership -----------------------------


def test_dataset_only_ranks_the_names_in_the_universe_that_session():
    universe = liquid_universe(30)
    params = DatasetParams(min_universe=5)
    u = build_universe(universe, UniverseParams(top_n=8, min_history=FULL_HISTORY))

    unrestricted = build_dataset(universe, params)
    restricted = build_dataset(universe, params, universe=u)

    assert len(set(unrestricted.symbols)) == 30
    assert set(restricted.symbols) == {f"S{k:02d}" for k in range(8)}
    assert restricted.n_samples < unrestricted.n_samples


# --- survivorship: the precondition, reported ------------------------------


def test_a_list_where_everyone_survives_is_named_as_one():
    """The headline case: today's ticker list. Every candidate spans the whole
    window, so there are no delistings and the selection mechanism, however
    correct, is picking among companies that all made it."""
    report = survivorship_report(liquid_universe(12))
    assert report["names_ending_early"] == 0
    assert report["names_spanning_everything"] == 12
    assert "SURVIVOR LIST" in report["verdict"]


def test_exits_are_detected_and_change_the_verdict():
    universe = liquid_universe(12)
    universe["S00"] = bars("S00", volume=5_000_000.0, n=150, seed=3)  # stops early
    universe["S01"] = bars("S01", volume=5_000_000.0, n=200, first_index=200, seed=4)  # lists late
    report = survivorship_report(universe)
    assert report["names_ending_early"] == 1
    assert report["names_entering_late"] == 1
    assert "SURVIVOR LIST" not in report["verdict"]
    assert "exits" in report["verdict"]


def test_a_short_data_gap_is_not_a_delisting():
    """Tolerance exists so a name that stops a few days early for data reasons
    is not reported as an exit — that would make the verdict unreadable."""
    universe = liquid_universe(6)
    universe["S00"] = bars("S00", volume=5_000_000.0, n=N - 10, seed=3)
    assert survivorship_report(universe)["names_ending_early"] == 0
