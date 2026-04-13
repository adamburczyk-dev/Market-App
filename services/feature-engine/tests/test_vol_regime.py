"""Tests for VolatilityRegimeCalculator — VIX-based exposure scaling."""

import pytest

from src.core.calculators.vol_regime import (
    EXPOSURE_SCALAR,
    classify_vix,
    exposure_scalar,
    target_vol_position_size,
)


class TestClassifyVix:
    def test_low(self):
        assert classify_vix(10) == "low"

    def test_low_boundary(self):
        assert classify_vix(0) == "low"

    def test_normal(self):
        assert classify_vix(17) == "normal"

    def test_normal_boundary(self):
        assert classify_vix(15) == "normal"

    def test_elevated(self):
        assert classify_vix(25) == "elevated"

    def test_high(self):
        assert classify_vix(35) == "high"

    def test_extreme(self):
        assert classify_vix(50) == "extreme"

    def test_extreme_boundary(self):
        assert classify_vix(40) == "extreme"

    def test_very_high_vix(self):
        assert classify_vix(120) == "extreme"


class TestExposureScalar:
    def test_low_vix_boosts(self):
        assert exposure_scalar(10) == pytest.approx(1.20)

    def test_normal_vix_neutral(self):
        assert exposure_scalar(17) == pytest.approx(1.00)

    def test_elevated_vix_reduces(self):
        assert exposure_scalar(25) == pytest.approx(0.70)

    def test_high_vix_significantly_reduces(self):
        assert exposure_scalar(35) == pytest.approx(0.40)

    def test_extreme_vix_minimal(self):
        assert exposure_scalar(50) == pytest.approx(0.15)

    def test_scalar_values_match_dict(self):
        for _regime, scalar in EXPOSURE_SCALAR.items():
            assert scalar > 0


class TestTargetVolPositionSize:
    def test_basic_calculation(self):
        # 100k * 0.15 / (100 * 0.30) = 15000 / 30 = 500
        shares = target_vol_position_size(100_000, 100.0, 0.30, target_vol=0.15)
        assert shares == 500

    def test_zero_vol_returns_zero(self):
        assert target_vol_position_size(100_000, 100.0, 0.0) == 0

    def test_zero_price_returns_zero(self):
        assert target_vol_position_size(100_000, 0.0, 0.30) == 0

    def test_high_vol_fewer_shares(self):
        low_vol = target_vol_position_size(100_000, 100.0, 0.20)
        high_vol = target_vol_position_size(100_000, 100.0, 0.60)
        assert low_vol > high_vol

    def test_result_is_int(self):
        shares = target_vol_position_size(100_000, 150.0, 0.25)
        assert isinstance(shares, int)
