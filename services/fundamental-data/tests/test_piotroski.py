"""Tests for the (partial) Piotroski F-Score."""

from datetime import date

from src.core.piotroski import compute_f_score

from .conftest import deteriorating_pair, improving_pair, stmt


def test_all_seven_signals_pass():
    current, prior = improving_pair()
    b = compute_f_score(current, prior)
    assert b.score == 7
    assert b.positive_net_income
    assert b.positive_operating_cash_flow
    assert b.quality_of_earnings
    assert b.improving_roa
    assert b.decreasing_leverage
    assert b.improving_net_margin
    assert b.improving_asset_turnover


def test_all_signals_fail():
    current, prior = deteriorating_pair()
    b = compute_f_score(current, prior)
    assert b.score == 0


def test_current_only_scores_max_three():
    current, _ = improving_pair()
    b = compute_f_score(current, prior=None)
    # only the three current-period signals can fire without a prior period
    assert b.score == 3
    assert b.improving_roa is False
    assert b.decreasing_leverage is False


def test_quality_of_earnings_requires_ocf_above_ni():
    # OCF < NI → accruals signal fails even though the firm is profitable
    current = stmt(date(2024, 12, 31), 1000, 300, 1000, 400, 100)
    b = compute_f_score(current, prior=None)
    assert b.positive_net_income is True
    assert b.quality_of_earnings is False


def test_missing_fields_fail_conservatively():
    current = stmt(date(2024, 12, 31))  # everything None
    b = compute_f_score(current, prior=None)
    assert b.score == 0


def test_zero_assets_does_not_crash():
    current = stmt(date(2024, 12, 31), 1000, 100, 0, 0, 120)
    prior = stmt(date(2023, 12, 31), 900, 80, 0, 0, 90)
    b = compute_f_score(current, prior)
    # ratios with a zero denominator are skipped (fail), no ZeroDivisionError
    assert b.improving_roa is False
    assert b.improving_asset_turnover is False
    assert b.positive_net_income is True


def test_omitted_criteria_documented():
    b = compute_f_score(*improving_pair())
    assert set(b.omitted) == {"current_ratio_change", "share_issuance"}
    assert b.as_dict()["max_score"] == 7
