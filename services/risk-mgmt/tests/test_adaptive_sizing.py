"""Tests for DrawdownAdaptiveSizer — anti-martingale position sizing."""

import pytest

from src.core.adaptive_sizing import DrawdownAdaptiveSizer


class TestConstructorValidation:
    def test_negative_base_risk_raises(self):
        with pytest.raises(ValueError, match="base_risk_per_trade"):
            DrawdownAdaptiveSizer(base_risk_per_trade=-0.01)

    def test_zero_base_risk_raises(self):
        with pytest.raises(ValueError, match="base_risk_per_trade"):
            DrawdownAdaptiveSizer(base_risk_per_trade=0.0)

    def test_negative_dd_start_raises(self):
        with pytest.raises(ValueError, match="dd_scaling_start"):
            DrawdownAdaptiveSizer(dd_scaling_start=-0.01)

    def test_start_gte_end_raises(self):
        with pytest.raises(ValueError, match="dd_scaling_start"):
            DrawdownAdaptiveSizer(dd_scaling_start=0.15, dd_scaling_end=0.10)

    def test_start_equals_end_raises(self):
        with pytest.raises(ValueError, match="dd_scaling_start"):
            DrawdownAdaptiveSizer(dd_scaling_start=0.10, dd_scaling_end=0.10)

    def test_negative_max_position_pct_raises(self):
        with pytest.raises(ValueError, match="max_position_pct"):
            DrawdownAdaptiveSizer(max_position_pct=-0.01)

    def test_custom_max_position_pct(self):
        sizer = DrawdownAdaptiveSizer(max_position_pct=0.10)
        # With 10% cap: 100k * 10% / 50 = 200 shares
        shares = sizer.position_size(100_000, 50.0, 49.0, 0.0)
        assert shares == 200


class TestComputeRiskBudget:
    def setup_method(self):
        self.sizer = DrawdownAdaptiveSizer()

    def test_no_drawdown_returns_base(self):
        assert self.sizer.compute_risk_budget(0.0) == 0.02

    def test_below_start_returns_base(self):
        assert self.sizer.compute_risk_budget(0.03) == 0.02

    def test_at_start_returns_base(self):
        assert self.sizer.compute_risk_budget(0.05) == 0.02

    def test_at_end_returns_zero(self):
        assert self.sizer.compute_risk_budget(0.15) == 0.0

    def test_beyond_end_returns_zero(self):
        assert self.sizer.compute_risk_budget(0.20) == 0.0

    def test_midpoint_returns_half(self):
        # dd=0.10 is midpoint between 0.05 and 0.15
        budget = self.sizer.compute_risk_budget(0.10)
        assert budget == pytest.approx(0.01, abs=1e-6)

    def test_quarter_point(self):
        # dd=0.075 → 25% through scaling range → 75% of base
        budget = self.sizer.compute_risk_budget(0.075)
        assert budget == pytest.approx(0.015, abs=1e-6)

    def test_three_quarter_point(self):
        # dd=0.125 → 75% through → 25% of base
        budget = self.sizer.compute_risk_budget(0.125)
        assert budget == pytest.approx(0.005, abs=1e-6)

    def test_negative_drawdown_uses_abs(self):
        budget = self.sizer.compute_risk_budget(-0.10)
        assert budget == pytest.approx(0.01, abs=1e-6)

    def test_custom_parameters(self):
        sizer = DrawdownAdaptiveSizer(
            base_risk_per_trade=0.03,
            dd_scaling_start=0.10,
            dd_scaling_end=0.20,
        )
        assert sizer.compute_risk_budget(0.0) == 0.03
        assert sizer.compute_risk_budget(0.15) == pytest.approx(0.015, abs=1e-6)
        assert sizer.compute_risk_budget(0.20) == 0.0

    def test_default_values_match_spec(self):
        sizer = DrawdownAdaptiveSizer()
        assert sizer.base_risk == 0.02
        assert sizer.dd_start == 0.05
        assert sizer.dd_end == 0.15


class TestPositionSize:
    def setup_method(self):
        self.sizer = DrawdownAdaptiveSizer()

    def test_basic_position_size(self):
        # dd=0% → risk=2% → max_risk=2000
        # entry=100, stop=90 → risk_per_share=10 → shares=200
        # cap: 100k * 5% / 100 = 50 → min(200, 50) = 50
        shares = self.sizer.position_size(100_000, 100.0, 90.0, 0.0)
        assert shares == 50

    def test_zero_risk_budget_returns_zero(self):
        shares = self.sizer.position_size(100_000, 150.0, 140.0, 0.20)
        assert shares == 0

    def test_zero_risk_per_share_returns_zero(self):
        shares = self.sizer.position_size(100_000, 150.0, 150.0, 0.0)
        assert shares == 0

    def test_capped_at_max_position(self):
        # Wide stop: risk allows many shares but 5% cap limits
        shares = self.sizer.position_size(100_000, 50.0, 49.0, 0.0)
        max_by_cap = int(100_000 * 0.05 / 50.0)  # 100
        assert shares == max_by_cap

    def test_fractional_shares_floored(self):
        shares = self.sizer.position_size(10_000, 150.0, 100.0, 0.0)
        assert isinstance(shares, int)

    def test_typical_scenario(self):
        """$100k portfolio, $150 entry, $140 stop, 3% drawdown."""
        shares = self.sizer.position_size(100_000, 150.0, 140.0, 0.03)
        # dd=3% < 5% start → full budget 2%
        # max_risk = 2000, risk_per_share = 10, shares_by_risk = 200
        # cap: 100k * 5% / 150 = 33
        assert shares == 33

    def test_reduced_at_drawdown(self):
        """At 10% drawdown, budget is halved → fewer shares."""
        # Use tight stop so risk constraint binds, not 5% cap
        # entry=100, stop=99 → risk_per_share=1
        # dd=0%: budget=2%, max_risk=2000, shares_by_risk=2000, cap=50 → 50
        # dd=10%: budget=1%, max_risk=1000, shares_by_risk=1000, cap=50 → 50
        # Still capped. Use higher price:
        # entry=500, stop=499, cap=100k*5%/500=10
        # dd=0%: shares_by_risk=2000, cap=10 → 10 (still cap)
        # Need risk to bind: entry=100, stop=50, risk=50
        # dd=0%: max_risk=2000, shares=40, cap=50 → 40
        # dd=10%: max_risk=1000, shares=20, cap=50 → 20
        shares_full = self.sizer.position_size(100_000, 100.0, 50.0, 0.0)
        shares_half = self.sizer.position_size(100_000, 100.0, 50.0, 0.10)
        assert shares_full == 40
        assert shares_half == 19  # 999.99.../50 floored by int()
