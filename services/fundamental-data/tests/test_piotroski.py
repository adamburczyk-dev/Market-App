"""Tests for the (partial) Piotroski F-Score."""

from datetime import date

from src.core.piotroski import compute_f_score

from .conftest import deteriorating_pair, improving_pair, stmt


def test_all_nine_signals_pass():
    current, prior = improving_pair()
    b = compute_f_score(current, prior)
    assert b.score == 9
    assert b.positive_net_income
    assert b.positive_operating_cash_flow
    assert b.quality_of_earnings
    assert b.improving_roa
    assert b.decreasing_leverage
    assert b.improving_current_ratio
    assert b.no_dilution
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


def test_max_score_is_nine():
    b = compute_f_score(*improving_pair())
    assert b.as_dict()["max_score"] == 9
    assert b.as_dict()["score"] == 9


def test_without_balance_sheet_detail_caps_at_seven():
    # legacy statements (no current assets/liabilities, no share count):
    # the liquidity + dilution signals fail conservatively → max effective 7
    current = stmt(date(2024, 12, 31), 1200, 200, 1000, 400, 250)
    prior = stmt(date(2023, 12, 31), 1000, 100, 1000, 500, 120)
    b = compute_f_score(current, prior)
    assert b.score == 7
    assert b.improving_current_ratio is False
    assert b.no_dilution is False


def test_flat_share_count_is_no_dilution():
    current = stmt(date(2024, 12, 31), 1200, 200, 1000, 400, 250, shares_outstanding=100)
    prior = stmt(date(2023, 12, 31), 1000, 100, 1000, 500, 120, shares_outstanding=100)
    b = compute_f_score(current, prior)
    assert b.no_dilution is True  # unchanged share count counts as no issuance


def test_zero_current_liabilities_skips_liquidity_signal():
    current = stmt(date(2024, 12, 31), 1200, 200, 1000, 400, 250, 500, 0)
    prior = stmt(date(2023, 12, 31), 1000, 100, 1000, 500, 120, 400, 250)
    b = compute_f_score(current, prior)
    assert b.improving_current_ratio is False  # degenerate denominator → conservative fail
