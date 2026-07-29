"""Fractional Kelly sizing (P5-3).

The rule is small; what needs pinning is that it cannot become a way to take
MORE risk than the existing envelope allows. A sizing change that can raise
exposure is a change to the risk limits wearing the costume of a model
improvement, and the non-negotiable rules (5% per position, 80% gross) are not
the model's to renegotiate.
"""

import pytest

from trading_common.sizing import (
    KellyParams,
    kelly_fraction,
    kelly_weight,
    scale_to_exposure,
)


def test_a_coin_flip_is_not_a_bet():
    """The anchor for the whole rule: at p = 0.5 the edge is zero, so the size
    is zero — not "small", and certainly not a fixed 2% risk budget."""
    assert kelly_fraction(0.5) == 0.0
    assert kelly_weight(0.5, envelope_weight=0.05) == 0.0


def test_the_fraction_is_linear_in_the_edge_for_symmetric_barriers():
    """With triple-barrier targets the payoff ratio is 1, so f* = 2p - 1."""
    assert kelly_fraction(0.60) == pytest.approx(0.20)
    assert kelly_fraction(0.75) == pytest.approx(0.50)
    assert kelly_fraction(1.0) == pytest.approx(1.0)


def test_an_unfavourable_probability_sizes_nothing_rather_than_shorting():
    """The book is long-only (R4). A negative Kelly fraction is not a short, it
    is a decision not to trade — returning the negative number would invite a
    caller to read it as a size."""
    assert kelly_fraction(0.40) == 0.0
    assert kelly_fraction(0.0) == 0.0
    assert kelly_weight(0.2, envelope_weight=0.05) == 0.0


def test_a_marginal_edge_is_treated_as_no_edge():
    """Near p = 0.5 the SIGN of the edge is estimation noise, and a book full of
    tiny noise-driven positions pays costs for nothing."""
    params = KellyParams(min_edge=0.02)
    assert kelly_fraction(0.505, params) == 0.0  # edge 0.01 < 0.02
    assert kelly_fraction(0.52, params) > 0.0


def test_the_asymmetric_payoff_case_uses_the_real_ratio():
    """A 2:1 payoff makes the same probability worth more. Passing the geometry
    explicitly is the alternative to quietly assuming it is symmetric."""
    symmetric = kelly_fraction(0.6, KellyParams(payoff_ratio=1.0))
    favourable = kelly_fraction(0.6, KellyParams(payoff_ratio=2.0))
    assert favourable > symmetric
    # (p*b - q)/b with p=0.6, b=2 → (1.2 - 0.4)/2 = 0.4
    assert favourable == pytest.approx(0.4)
    assert kelly_fraction(0.6, KellyParams(payoff_ratio=0.0)) == 0.0


def test_fractional_kelly_bets_a_fraction_of_the_optimum():
    """Full Kelly is optimal only under a perfectly known probability, and its
    downside is asymmetric — overbetting 2x turns a growth-optimal bet into a
    zero-growth one. A quarter is the default for that reason."""
    quarter = kelly_weight(0.6, 1.0, KellyParams(fraction=0.25, max_position_weight=1.0))
    full = kelly_weight(0.6, 1.0, KellyParams(fraction=1.0, max_position_weight=1.0))
    assert quarter == pytest.approx(0.05)  # 0.25 * 0.20
    assert full == pytest.approx(0.20)
    assert quarter == pytest.approx(full / 4)


def test_the_envelope_always_wins_however_confident_the_model():
    """The safety argument. Enabling Kelly may only ever SHRINK a position
    relative to the rules already in force — a certain model with a drawdown-
    throttled envelope still gets the throttled size."""
    certain = KellyParams(fraction=1.0, max_position_weight=1.0)
    assert kelly_weight(0.99, envelope_weight=0.004, params=certain) == pytest.approx(0.004)
    assert kelly_weight(0.99, envelope_weight=0.0, params=certain) == 0.0
    # ...and the 5% per-position rule binds even with a generous envelope
    assert kelly_weight(0.99, envelope_weight=0.50, params=KellyParams(fraction=1.0)) == 0.05


def test_a_negative_envelope_is_not_a_short_either():
    assert kelly_weight(0.9, envelope_weight=-0.10) == 0.0


def test_kelly_sizes_each_bet_as_if_it_were_the_only_one_so_the_book_is_capped():
    """Ten confident positions at 5% is a 50% book and twenty is 100% — Kelly
    has no view on that, because it prices one bet at a time. The gross cap is
    what stops it."""
    weights = {f"S{i}": 0.05 for i in range(20)}
    scaled = scale_to_exposure(weights, max_exposure=0.80)
    assert sum(scaled.values()) == pytest.approx(0.80)
    assert all(w == pytest.approx(0.04) for w in scaled.values())


def test_scaling_preserves_relative_conviction():
    """Proportional scaling, not truncation: the model said one name deserved
    twice another, and cutting the tail would silently overrule that."""
    weights = {"A": 0.05, "B": 0.025, "C": 0.05, "D": 0.05}
    scaled = scale_to_exposure(weights, max_exposure=0.10)
    assert sum(scaled.values()) == pytest.approx(0.10)
    assert scaled["A"] == pytest.approx(2 * scaled["B"])
    assert set(scaled) == set(weights), "a capped book must not silently drop names"


def test_a_book_within_the_cap_is_left_alone():
    weights = {"A": 0.05, "B": 0.03}
    assert scale_to_exposure(weights, max_exposure=0.80) == weights
    assert scale_to_exposure({}, max_exposure=0.80) == {}
