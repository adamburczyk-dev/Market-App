"""Tests for CrossAssetMomentumCalculator — inter-market risk appetite."""

import pytest

from src.core.calculators.cross_asset import compute_cross_asset_scores


class TestComputeCrossAssetScores:
    def test_basic_relative_strength(self):
        result = compute_cross_asset_scores(
            {"AAPL": 0.10, "MSFT": 0.08},
            spy_return_60d=0.05,
        )
        rs = result["relative_strength"]
        assert rs["AAPL"] == pytest.approx(0.05)
        assert rs["MSFT"] == pytest.approx(0.03)

    def test_risk_appetite_positive(self):
        """Assets outperforming SPY → positive risk appetite."""
        result = compute_cross_asset_scores(
            {"A": 0.10, "B": 0.08},
            spy_return_60d=0.05,
        )
        assert result["risk_appetite_score"] > 0

    def test_risk_appetite_negative(self):
        """Assets underperforming SPY → negative risk appetite."""
        result = compute_cross_asset_scores(
            {"A": 0.02, "B": 0.01},
            spy_return_60d=0.05,
        )
        assert result["risk_appetite_score"] < 0

    def test_empty_assets(self):
        result = compute_cross_asset_scores({}, spy_return_60d=0.05)
        assert result["relative_strength"] == {}
        assert result["risk_appetite_score"] == 0.0

    def test_single_asset(self):
        result = compute_cross_asset_scores(
            {"AAPL": 0.15},
            spy_return_60d=0.10,
        )
        assert result["risk_appetite_score"] == pytest.approx(0.05)

    def test_zero_spy_return(self):
        result = compute_cross_asset_scores(
            {"A": 0.05, "B": -0.03},
            spy_return_60d=0.0,
        )
        rs = result["relative_strength"]
        assert rs["A"] == pytest.approx(0.05)
        assert rs["B"] == pytest.approx(-0.03)

    def test_risk_appetite_is_mean_of_relative_strengths(self):
        result = compute_cross_asset_scores(
            {"A": 0.10, "B": 0.06, "C": 0.02},
            spy_return_60d=0.04,
        )
        # RS: 0.06, 0.02, -0.02 → mean = 0.02
        assert result["risk_appetite_score"] == pytest.approx(0.02)

    def test_all_equal_returns(self):
        result = compute_cross_asset_scores(
            {"A": 0.05, "B": 0.05},
            spy_return_60d=0.05,
        )
        assert result["risk_appetite_score"] == pytest.approx(0.0)
