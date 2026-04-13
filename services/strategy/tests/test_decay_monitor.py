"""Tests for StrategyDecayMonitor — detect degrading strategies."""

from src.core.decay_monitor import StrategyDecayMonitor

# Healthy defaults: all above active thresholds
HEALTHY = {
    "strategy_name": "momentum_v1",
    "sharpe_30d": 1.0,
    "sharpe_90d": 0.8,
    "sharpe_180d": 0.7,
    "win_rate_30d": 0.55,
    "profit_factor_30d": 1.5,
    "excess_return_vs_spy_30d": 0.02,
}


class TestActiveStatus:
    def setup_method(self):
        self.monitor = StrategyDecayMonitor()

    def test_healthy_strategy_is_active(self):
        health = self.monitor.evaluate(**HEALTHY)
        assert health.status == "active"
        assert health.reason == "all_metrics_healthy"

    def test_at_exact_active_thresholds(self):
        health = self.monitor.evaluate(
            strategy_name="test",
            sharpe_30d=0.5,
            sharpe_90d=0.5,
            sharpe_180d=0.5,
            win_rate_30d=0.4,
            profit_factor_30d=1.2,
            excess_return_vs_spy_30d=0.0,
        )
        assert health.status == "active"

    def test_health_fields_populated(self):
        health = self.monitor.evaluate(**HEALTHY)
        assert health.strategy_name == "momentum_v1"
        assert health.sharpe_30d == 1.0
        assert health.check_date is not None


class TestDeactivatedStatus:
    def setup_method(self):
        self.monitor = StrategyDecayMonitor()

    def test_negative_sharpe_deactivates(self):
        health = self.monitor.evaluate(**{**HEALTHY, "sharpe_30d": -0.5})
        assert health.status == "deactivated"
        assert health.reason == "negative_sharpe"

    def test_sharpe_exactly_zero_not_deactivated(self):
        """Sharpe == 0 is NOT < 0, so not deactivated by sharpe rule."""
        health = self.monitor.evaluate(**{**HEALTHY, "sharpe_30d": 0.0})
        # Not deactivated by sharpe, but below active threshold → probation
        assert health.status == "probation"

    def test_low_profit_factor_deactivates(self):
        health = self.monitor.evaluate(**{**HEALTHY, "profit_factor_30d": 0.7})
        assert health.status == "deactivated"
        assert health.reason == "low_profit_factor"

    def test_pf_exactly_at_deactivation_threshold(self):
        """PF < 0.8 deactivates; PF == 0.8 does not."""
        health = self.monitor.evaluate(**{**HEALTHY, "profit_factor_30d": 0.8})
        assert health.status != "deactivated"

    def test_probation_timeout_deactivates(self):
        health = self.monitor.evaluate(**HEALTHY, days_in_probation=31)
        assert health.status == "deactivated"
        assert health.reason == "probation_timeout"

    def test_probation_exactly_30_days_not_deactivated(self):
        """30 days is NOT > 30, so should not deactivate."""
        health = self.monitor.evaluate(**HEALTHY, days_in_probation=30)
        assert health.status != "deactivated"

    def test_negative_sharpe_checked_before_low_pf(self):
        """Both triggers present — negative sharpe takes priority."""
        health = self.monitor.evaluate(**{**HEALTHY, "sharpe_30d": -1.0, "profit_factor_30d": 0.5})
        assert health.reason == "negative_sharpe"


class TestProbationStatus:
    def setup_method(self):
        self.monitor = StrategyDecayMonitor()

    def test_low_sharpe_causes_probation(self):
        health = self.monitor.evaluate(**{**HEALTHY, "sharpe_30d": 0.3})
        assert health.status == "probation"
        assert "low_sharpe" in health.reason

    def test_low_win_rate_causes_probation(self):
        health = self.monitor.evaluate(**{**HEALTHY, "win_rate_30d": 0.35})
        assert health.status == "probation"
        assert "low_win_rate" in health.reason

    def test_low_pf_causes_probation(self):
        """PF between 0.8 and 1.2 → probation (not deactivated, not active)."""
        health = self.monitor.evaluate(**{**HEALTHY, "profit_factor_30d": 1.0})
        assert health.status == "probation"
        assert "low_pf" in health.reason

    def test_multiple_probation_reasons(self):
        health = self.monitor.evaluate(**{**HEALTHY, "sharpe_30d": 0.3, "win_rate_30d": 0.35})
        assert health.status == "probation"
        assert "low_sharpe" in health.reason
        assert "low_win_rate" in health.reason


class TestThresholdConstants:
    def test_active_sharpe_min(self):
        assert StrategyDecayMonitor.ACTIVE_SHARPE_MIN == 0.5

    def test_active_pf_min(self):
        assert StrategyDecayMonitor.ACTIVE_PF_MIN == 1.2

    def test_active_wr_min(self):
        assert StrategyDecayMonitor.ACTIVE_WR_MIN == 0.4

    def test_deactivate_sharpe_max(self):
        assert StrategyDecayMonitor.DEACTIVATE_SHARPE_MAX == 0.0

    def test_deactivate_pf_max(self):
        assert StrategyDecayMonitor.DEACTIVATE_PF_MAX == 0.8

    def test_probation_max_days(self):
        assert StrategyDecayMonitor.PROBATION_MAX_DAYS == 30
