"""Tests for the weighted signal combination."""

from src.core.aggregator import combine

from .conftest import components

EQUAL = {"strategy": 1 / 3, "ml": 1 / 3, "macro": 1 / 3}


def test_empty_is_hold():
    signal, conf, score = combine([], EQUAL)
    assert signal == "HOLD"
    assert conf == 0.0


def test_consensus_buy():
    signal, conf, score = combine(
        components(("strategy", "BUY", 0.9), ("ml", "BUY", 0.8), ("macro", "BUY", 0.7)), EQUAL
    )
    assert signal == "BUY"
    assert conf > 0.2
    assert score > 0


def test_consensus_sell():
    signal, _, score = combine(components(("strategy", "SELL", 0.9), ("ml", "SELL", 0.8)), EQUAL)
    assert signal == "SELL"
    assert score < 0


def test_conflict_nets_to_hold():
    signal, conf, score = combine(components(("strategy", "BUY", 0.8), ("ml", "SELL", 0.8)), EQUAL)
    assert signal == "HOLD"
    assert score == 0.0


def test_below_threshold_is_hold():
    # single weak BUY under equal weights → score = (1/3)*0.3 = 0.1 < 0.2 threshold
    signal, _, score = combine(components(("strategy", "BUY", 0.3)), EQUAL)
    assert signal == "HOLD"
    assert 0 < score < 0.2


def test_weights_shift_the_decision():
    comps = components(("strategy", "BUY", 0.9), ("ml", "SELL", 0.5))
    # strategy dominates → BUY
    heavy_strategy = {"strategy": 0.8, "ml": 0.2, "macro": 0.0}
    assert combine(comps, heavy_strategy)[0] == "BUY"
    # ml dominates → SELL
    heavy_ml = {"strategy": 0.2, "ml": 0.8, "macro": 0.0}
    assert combine(comps, heavy_ml)[0] == "SELL"


def test_hold_components_are_neutral():
    signal, _, score = combine(components(("strategy", "BUY", 0.9), ("ml", "HOLD", 0.9)), EQUAL)
    # only strategy contributes: (1/3)*0.9 = 0.3 ≥ 0.2 → BUY
    assert signal == "BUY"


def test_confidence_capped_at_one():
    signal, conf, score = combine(
        components(("strategy", "BUY", 1.0), ("ml", "BUY", 1.0), ("macro", "BUY", 1.0)),
        {"strategy": 1.0, "ml": 1.0, "macro": 1.0},  # unnormalized → score 3.0
    )
    assert conf == 1.0
