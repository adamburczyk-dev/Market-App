"""Tests for CostAwareFilter — transaction cost filtering."""

import pytest

from src.core.cost_filter import CAP_TIER_MULTIPLIERS, CostAwareFilter, TransactionCosts


class TestTransactionCosts:
    def test_default_spread(self):
        assert TransactionCosts().spread_bps == 5.0

    def test_default_slippage(self):
        assert TransactionCosts().slippage_bps == 5.0

    def test_default_market_impact(self):
        assert TransactionCosts().market_impact_bps == 2.0

    def test_roundtrip_default(self):
        """2 * (5 + 5 + 2) = 24 bps."""
        assert TransactionCosts().total_roundtrip_bps == pytest.approx(24.0)

    def test_roundtrip_custom(self):
        costs = TransactionCosts(spread_bps=10, slippage_bps=8, market_impact_bps=5)
        assert costs.total_roundtrip_bps == pytest.approx(46.0)


class TestCapTierMultipliers:
    def test_large(self):
        assert CAP_TIER_MULTIPLIERS["large"] == 1.0

    def test_mid(self):
        assert CAP_TIER_MULTIPLIERS["mid"] == 1.5

    def test_small(self):
        assert CAP_TIER_MULTIPLIERS["small"] == 2.5

    def test_micro(self):
        assert CAP_TIER_MULTIPLIERS["micro"] == 5.0


class TestCostAwareFilterProfitable:
    def setup_method(self):
        self.filt = CostAwareFilter()

    def test_large_cap_profitable(self):
        """Edge 100 bps > required 48 bps (24 * 2) → profitable."""
        ok, details = self.filt.is_profitable_after_costs(100.0, market_cap_tier="large")
        assert ok is True
        assert details["adjusted_cost_bps"] == pytest.approx(24.0)
        assert details["required_edge_bps"] == pytest.approx(48.0)

    def test_exactly_at_required_edge(self):
        """Edge == required → profitable (>= not >)."""
        ok, _ = self.filt.is_profitable_after_costs(48.0, market_cap_tier="large")
        assert ok is True

    def test_multi_day_hold_profitable(self):
        """Holding period doesn't change profitability threshold — cost is per trade."""
        ok, details = self.filt.is_profitable_after_costs(
            100.0, holding_period_days=5, market_cap_tier="large"
        )
        assert ok is True
        assert details["cost_per_day_bps"] == pytest.approx(24.0 / 5)


class TestCostAwareFilterUnprofitable:
    def setup_method(self):
        self.filt = CostAwareFilter()

    def test_large_cap_unprofitable(self):
        """Edge 30 bps < required 48 bps → unprofitable."""
        ok, _ = self.filt.is_profitable_after_costs(30.0, market_cap_tier="large")
        assert ok is False

    def test_mid_cap_higher_threshold(self):
        """Mid cap: required = 24 * 1.5 * 2 = 72 bps."""
        ok, details = self.filt.is_profitable_after_costs(50.0, market_cap_tier="mid")
        assert ok is False
        assert details["required_edge_bps"] == pytest.approx(72.0)

    def test_small_cap_even_higher(self):
        """Small cap: required = 24 * 2.5 * 2 = 120 bps."""
        ok, details = self.filt.is_profitable_after_costs(100.0, market_cap_tier="small")
        assert ok is False
        assert details["required_edge_bps"] == pytest.approx(120.0)

    def test_micro_cap_very_expensive(self):
        """Micro cap: required = 24 * 5.0 * 2 = 240 bps."""
        ok, details = self.filt.is_profitable_after_costs(200.0, market_cap_tier="micro")
        assert ok is False
        assert details["required_edge_bps"] == pytest.approx(240.0)


class TestCostAwareFilterDetails:
    def setup_method(self):
        self.filt = CostAwareFilter()

    def test_edge_to_cost_ratio(self):
        _, details = self.filt.is_profitable_after_costs(72.0, market_cap_tier="large")
        assert details["edge_to_cost_ratio"] == pytest.approx(3.0)

    def test_unknown_cap_tier_uses_default(self):
        """Unknown tier defaults to multiplier 1.0."""
        _, details = self.filt.is_profitable_after_costs(100.0, market_cap_tier="unknown")
        assert details["adjusted_cost_bps"] == pytest.approx(24.0)

    def test_custom_min_edge_multiple(self):
        filt = CostAwareFilter(min_edge_multiple=3.0)
        ok, details = filt.is_profitable_after_costs(60.0, market_cap_tier="large")
        # required = 24 * 3 = 72 > 60
        assert ok is False
        assert details["required_edge_bps"] == pytest.approx(72.0)

    def test_custom_costs(self):
        costs = TransactionCosts(spread_bps=10, slippage_bps=10, market_impact_bps=5)
        filt = CostAwareFilter(costs=costs)
        _, details = filt.is_profitable_after_costs(100.0, market_cap_tier="large")
        # roundtrip = 2 * 25 = 50, required = 50 * 2 = 100
        assert details["adjusted_cost_bps"] == pytest.approx(50.0)
