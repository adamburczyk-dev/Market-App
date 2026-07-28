"""Testy cross-sectional percentile ranking."""

from datetime import UTC, datetime

import pytest

from trading_common.ranking import cross_sectional_rank, sector_neutralize
from trading_common.schemas import FeatureVector, Interval


def _fv(symbol: str, **features: float) -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        interval=Interval.D1,
        features=features,
        tier=1,
    )


def test_ranks_lowest_to_highest():
    ranked = {
        v.symbol: v
        for v in cross_sectional_rank(
            [_fv("A", momentum=0.1), _fv("B", momentum=0.5), _fv("C", momentum=0.9)]
        )
    }
    assert ranked["A"].features["momentum"] == 0.0
    assert ranked["B"].features["momentum"] == 0.5
    assert ranked["C"].features["momentum"] == 1.0
    assert all(v.rank_transformed for v in ranked.values())


def test_single_symbol_is_neutral():
    ranked = cross_sectional_rank([_fv("A", x=42.0)])
    assert ranked[0].features["x"] == 0.5
    assert ranked[0].rank_transformed is True


def test_ties_share_mean_rank():
    r = {
        v.symbol: v.features["x"]
        for v in cross_sectional_rank([_fv("A", x=1.0), _fv("B", x=1.0), _fv("C", x=2.0)])
    }
    assert r["A"] == 0.25  # A,B tie at the bottom (mean of ranks 0,1 → 0.5/2)
    assert r["B"] == 0.25
    assert r["C"] == 1.0


def test_per_feature_independent_and_missing_keys():
    r = {
        v.symbol: v.features
        for v in cross_sectional_rank(
            [_fv("A", x=1.0, y=9.0), _fv("B", x=2.0)]  # B lacks y
        )
    }
    assert r["A"]["x"] == 0.0
    assert r["B"]["x"] == 1.0
    assert r["A"]["y"] == 0.5  # only A has y -> single-value -> neutral
    assert "y" not in r["B"]


def test_inputs_left_unchanged():
    vs = [_fv("A", x=1.0), _fv("B", x=2.0)]
    cross_sectional_rank(vs)
    assert vs[0].features["x"] == 1.0
    assert vs[0].rank_transformed is False


def test_empty_universe():
    assert cross_sectional_rank([]) == []


# --- P2-2: sector neutralization -----------------------------------------


def universe(**by_sector: list[tuple[str, float]]) -> tuple[list, dict[str, str]]:
    """(vectors, sector map) from {sector: [(symbol, momentum), ...]}."""
    vectors, sectors = [], {}
    for sector, members in by_sector.items():
        name = sector.replace("_", " ").title()
        for symbol, momentum in members:
            vectors.append(_fv(symbol, momentum=momentum))
            sectors[symbol] = name
    return vectors, sectors


def test_neutralization_removes_the_sector_level():
    """The failure it exists to prevent: one sector runs, and a global rank
    reads that as every name in it being strong.

    Energy sits far above Utilities on the raw value, so a global rank puts all
    four Energy names on top. After demeaning, what survives is each name's
    position among its OWN peers — and the two sectors interleave.
    """
    vectors, sectors = universe(
        energy=[("E1", 10.0), ("E2", 11.0), ("E3", 12.0), ("E4", 13.0)],
        utilities=[("U1", 1.0), ("U2", 2.0), ("U3", 3.0), ("U4", 4.0)],
    )
    plain = {v.symbol: v.features["momentum"] for v in cross_sectional_rank(vectors)}
    assert min(plain[s] for s in ("E1", "E2", "E3", "E4")) > max(
        plain[s] for s in ("U1", "U2", "U3", "U4")
    )

    neutral = {
        v.symbol: v.features["momentum"]
        for v in cross_sectional_rank(sector_neutralize(vectors, sectors))
    }
    # E1 is the weakest energy name and U4 the strongest utility: after
    # neutralizing, the utility outranks the energy name.
    assert neutral["U4"] > neutral["E1"]
    assert neutral["E4"] == neutral["U4"]  # each is top of its own sector


def test_undersized_sectors_are_pooled_not_zeroed():
    """A one-name sector has itself as its median. Demeaning it against itself
    would set every feature to 0 — a fabricated 'exactly average' — so those
    names are pooled into one residual group instead."""
    vectors, sectors = universe(
        energy=[("E1", 10.0), ("E2", 11.0), ("E3", 12.0), ("E4", 13.0)],
        materials=[("M1", 5.0)],
        real_estate=[("R1", 7.0)],
        utilities=[("U1", 1.0), ("U2", 2.0)],
    )
    out = {v.symbol: v.features["momentum"] for v in sector_neutralize(vectors, sectors)}
    assert out["M1"] != 0.0  # not demeaned against itself
    # M1, R1, U1, U2 form the residual group; its median is (5+7+1+2)/2 → 3.5
    assert out["U1"] == pytest.approx(1.0 - 3.5)
    assert out["M1"] == pytest.approx(5.0 - 3.5)
    assert out["E1"] == pytest.approx(10.0 - 11.5)  # energy median


def test_unknown_and_missing_sectors_join_the_residual_group():
    vectors, sectors = universe(energy=[("E1", 10.0), ("E2", 11.0), ("E3", 12.0), ("E4", 13.0)])
    vectors += [_fv("X1", momentum=1.0), _fv("X2", momentum=2.0)]
    vectors += [_fv("X3", momentum=3.0), _fv("X4", momentum=4.0)]
    sectors["X1"] = "Crypto"  # not a GICS sector
    # X2, X3, X4 are absent from the map entirely
    out = {v.symbol: v.features["momentum"] for v in sector_neutralize(vectors, sectors)}
    assert out["X1"] == pytest.approx(1.0 - 2.5)  # median of 1,2,3,4
    assert out["X4"] == pytest.approx(4.0 - 2.5)


def test_aliases_group_with_their_canonical_sector():
    """ "Technology" and "Information Technology" are one peer group, not two
    groups of two that both fall below the size threshold."""
    vectors = [
        _fv("A", momentum=1.0),
        _fv("B", momentum=2.0),
        _fv("C", momentum=3.0),
        _fv("D", momentum=4.0),
    ]
    mixed = {
        "A": "Technology",
        "B": "Information Technology",
        "C": "Tech",
        "D": "INFORMATION TECHNOLOGY",
    }
    out = {v.symbol: v.features["momentum"] for v in sector_neutralize(vectors, mixed)}
    assert out["A"] == pytest.approx(1.0 - 2.5)  # one group of four, median 2.5


def test_neutralization_leaves_inputs_alone_and_handles_empty():
    vectors, sectors = universe(energy=[("E1", 10.0), ("E2", 11.0), ("E3", 12.0), ("E4", 13.0)])
    sector_neutralize(vectors, sectors)
    assert vectors[0].features["momentum"] == 10.0
    assert sector_neutralize([], {}) == []
