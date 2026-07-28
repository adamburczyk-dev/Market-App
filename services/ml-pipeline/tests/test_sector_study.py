"""The E2 measurement for sector neutralization (P2-2).

Two things have to hold for this study to be worth acting on: it must detect a
sector effect when one is planted, and it must SAY SO when the universe is too
narrow for the transform to have applied — because "no improvement" and "barely
ran" are the same numbers and opposite conclusions.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
from trading_common.schemas import Interval, OHLCVBar

from src.core.dataset import DatasetParams
from src.core.sector_study import run_sector_study, sector_composition

START = datetime(2021, 1, 4, tzinfo=UTC)
SECTORS = ("Information Technology", "Health Care", "Financials", "Energy", "Utilities")


def bars(symbol: str, closes: np.ndarray) -> list[OHLCVBar]:
    return [
        OHLCVBar(
            symbol=symbol,
            timestamp=START + timedelta(days=i),
            interval=Interval.D1,
            open=float(c),
            high=float(c) * 1.01,
            low=float(c) * 0.99,
            close=float(c),
            adj_close=float(c),
            volume=1_000_000.0 + i,
            source="test",
        )
        for i, c in enumerate(closes)
    ]


def sector_universe(
    n_sectors: int = 5, per_sector: int = 6, n: int = 400, seed: int = 3
) -> tuple[dict[str, list[OHLCVBar]], dict[str, str | None]]:
    """A universe where the SECTOR carries the level and names differ within it.

    Each sector gets its own drift; a name is its sector's path times its own
    idiosyncratic wobble. That is the structure global ranking confuses with
    stock selection.
    """
    rng = np.random.default_rng(seed)
    universe: dict[str, list[OHLCVBar]] = {}
    sectors: dict[str, str | None] = {}
    for s in range(n_sectors):
        drift = 0.0012 * (s - n_sectors / 2)
        common = np.cumprod(1.0 + rng.normal(drift, 0.010, n))
        for k in range(per_sector):
            symbol = f"S{s}{k}"
            own = np.cumprod(1.0 + rng.normal(0.0, 0.006, n))
            universe[symbol] = bars(symbol, 100.0 * common * own)
            sectors[symbol] = SECTORS[s]
    return universe, sectors


PARAMS = DatasetParams(min_universe=20)


def test_composition_names_what_actually_got_neutralized():
    _, sectors = sector_universe(n_sectors=5, per_sector=6)
    comp = sector_composition(sorted(sectors), sectors)
    assert comp["symbols"] == 30
    assert len(comp["sectors_large_enough"]) == 5
    assert comp["names_in_residual_group"] == 0
    assert comp["share_neutralized_against_peers"] == 1.0


def test_composition_flags_a_universe_too_narrow_to_measure():
    """The real 34-name universe: 11 sectors, ~3 names each. Almost nothing has
    a peer group, so the study must report that rather than let the comparison
    be read as a verdict on the transform."""
    sectors: dict[str, str | None] = {}
    for s in range(11):
        for k in range(3):
            sectors[f"T{s}{k}"] = SECTORS[s % len(SECTORS)] if s < 3 else f"Sector{s}"
    comp = sector_composition(sorted(sectors), sectors)
    assert comp["share_neutralized_against_peers"] < 0.5
    assert comp["names_in_residual_group"] > len(sectors) / 2
    assert comp["unknown_sector"], "made-up sector names must be reported, not counted"


def test_study_compares_both_datasets_on_the_same_features():
    universe, sectors = sector_universe()
    result = run_sector_study(universe, sectors, PARAMS)

    assert result["rows_plain"] == result["rows_neutral"] > 0
    assert result["composition"]["share_neutralized_against_peers"] == 1.0
    names = [row["feature"] for row in result["features"]]
    assert "momentum_12_1" in names
    assert not any(n.startswith("macro_") for n in names)  # constant one-hots excluded
    assert result["summary"]["n_features"] == len(names)
    assert "no trials were consumed" in result["note"]
    # ordered by the evidence that survives demeaning — the actionable column
    surviving = [abs(row["t_neutral"]) for row in result["features"]]
    assert surviving == sorted(surviving, reverse=True)


def test_neutralization_actually_changes_the_ranks():
    """If the two datasets were identical the comparison would be a no-op that
    always reads 'no gain' — pin that the transform reaches the model input."""
    universe, sectors = sector_universe()
    result = run_sector_study(universe, sectors, PARAMS)
    changed = [row for row in result["features"] if abs(row["ic_plain"] - row["ic_neutral"]) > 1e-9]
    assert changed, "sector neutralization left every feature's IC untouched"


def test_a_single_sector_universe_leaves_the_study_a_no_op():
    """Every name in one sector: demeaning shifts the whole cross-section by a
    constant, and a percentile rank is invariant to that. The ICs must be
    identical — a useful sanity check that the transform is a within-group
    centring, not an accidental rescaling."""
    universe, sectors = sector_universe(n_sectors=1, per_sector=24)
    result = run_sector_study(universe, sectors, PARAMS)
    for row in result["features"]:
        assert row["ic_plain"] == row["ic_neutral"]
        assert row["t_gain"] == 0.0


def test_a_sector_bet_is_named_as_one():
    """The reading that matters, pinned on a universe built to have it.

    Sectors drift apart and names wobble around their sector, so a GLOBAL rank
    predicts strongly — it is reading the sector, which really does persist.
    Demeaning removes exactly that, and the evidence goes with it. A drop here
    is the study working, not the transform failing, and the report must not be
    able to present it as "no gain, keep ranking globally".
    """
    universe, sectors = sector_universe()
    result = run_sector_study(universe, sectors, PARAMS)
    summary = result["summary"]

    assert summary["strong_plain"] >= 5, "the fixture must have a strong global signal"
    assert summary["strong_neutral"] == 0, "none of it should survive demeaning"
    assert summary["mean_abs_t_neutral"] < summary["mean_abs_t_plain"]
    assert "was the sector" in result["note"] or "the sector" in result["note"]
