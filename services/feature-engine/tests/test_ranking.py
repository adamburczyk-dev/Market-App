"""Testy cross-sectional percentile ranking."""

from datetime import UTC, datetime

from trading_common.schemas import FeatureVector, Interval

from src.core.ranking import cross_sectional_rank


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
