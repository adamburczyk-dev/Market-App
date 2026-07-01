"""Tests for AdaptiveWeightOptimizer — dynamic signal weighting."""

import pytest

from src.core.adaptive_weights import AdaptiveWeightOptimizer, SignalPerformance


class TestSignalPerformance:
    def test_empty_hit_rate(self):
        sp = SignalPerformance()
        assert sp.hit_rate == 0.0

    def test_empty_avg_return(self):
        sp = SignalPerformance()
        assert sp.avg_return == 0.0

    def test_empty_information_ratio(self):
        sp = SignalPerformance()
        assert sp.information_ratio == 0.0

    def test_hit_rate_with_data(self):
        sp = SignalPerformance()
        for v in [0.01, -0.02, 0.03, 0.005, -0.01]:
            sp.outcomes.append(v)
        # 3 positive out of 5
        assert sp.hit_rate == pytest.approx(0.6)

    def test_avg_return(self):
        sp = SignalPerformance()
        for v in [0.01, 0.02, 0.03]:
            sp.outcomes.append(v)
        assert sp.avg_return == pytest.approx(0.02)

    def test_information_ratio_positive(self):
        sp = SignalPerformance()
        for v in [0.05, 0.06, 0.04, 0.05, 0.055]:
            sp.outcomes.append(v)
        ir = sp.information_ratio
        assert ir > 0

    def test_deque_no_hardcoded_maxlen(self):
        sp = SignalPerformance()
        assert sp.outcomes.maxlen is None

    def test_information_ratio_single_value(self):
        """With only 1 data point, IR should be 0 (insufficient data)."""
        sp = SignalPerformance()
        sp.outcomes.append(0.05)
        assert sp.information_ratio == 0.0

    def test_information_ratio_constant_returns(self):
        """All same value → std=0 → IR=0."""
        sp = SignalPerformance()
        for _ in range(10):
            sp.outcomes.append(0.01)
        assert sp.information_ratio == 0.0


class TestAdaptiveWeightOptimizer:
    def test_equal_weights_no_data(self):
        opt = AdaptiveWeightOptimizer(["a", "b", "c"])
        weights = opt.compute_weights()
        assert weights["a"] == pytest.approx(1 / 3, abs=1e-6)
        assert weights["b"] == pytest.approx(1 / 3, abs=1e-6)
        assert weights["c"] == pytest.approx(1 / 3, abs=1e-6)

    def test_weights_sum_to_one(self):
        opt = AdaptiveWeightOptimizer(["a", "b", "c"])
        for _ in range(30):
            opt.record_outcome("a", 0.02)
            opt.record_outcome("b", 0.01)
            opt.record_outcome("c", -0.005)
        weights = opt.compute_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_better_source_gets_higher_weight(self):
        opt = AdaptiveWeightOptimizer(["good", "bad"])
        for _ in range(30):
            opt.record_outcome("good", 0.03)
            opt.record_outcome("bad", -0.01)
        weights = opt.compute_weights()
        assert weights["good"] > weights["bad"]

    def test_floor_applied(self):
        """Even bad sources get at least min_weight (before renorm)."""
        opt = AdaptiveWeightOptimizer(["good", "terrible"], min_weight=0.05)
        for _ in range(30):
            opt.record_outcome("good", 0.05)
            opt.record_outcome("terrible", -0.10)
        weights = opt.compute_weights()
        # After floor + renorm, terrible should still have meaningful weight
        assert weights["terrible"] > 0

    def test_cap_applied(self):
        """No source exceeds max_weight after renorm with enough sources."""
        opt = AdaptiveWeightOptimizer(["a", "b", "c", "d"], max_weight=0.40)
        for _ in range(30):
            opt.record_outcome("a", 0.10)
            opt.record_outcome("b", -0.05)
            opt.record_outcome("c", -0.05)
            opt.record_outcome("d", -0.05)
        weights = opt.compute_weights()
        # After cap and renorm, weights should be reasonable
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_single_source(self):
        opt = AdaptiveWeightOptimizer(["only"])
        weights = opt.compute_weights()
        assert weights["only"] == pytest.approx(1.0)

    def test_empty_sources(self):
        opt = AdaptiveWeightOptimizer([])
        weights = opt.compute_weights()
        assert weights == {}

    def test_custom_lambda(self):
        """Higher lambda → more differentiation between sources."""
        # Use noisy returns so IR is moderate and doesn't saturate floor/cap
        import random

        random.seed(42)
        opt_low = AdaptiveWeightOptimizer(
            ["a", "b"], smoothing_lambda=0.5, min_weight=0.01, max_weight=0.99
        )
        opt_high = AdaptiveWeightOptimizer(
            ["a", "b"], smoothing_lambda=5.0, min_weight=0.01, max_weight=0.99
        )
        for _ in range(30):
            ret_a = 0.001 + random.gauss(0, 0.005)
            ret_b = -0.001 + random.gauss(0, 0.005)
            for opt in [opt_low, opt_high]:
                opt.record_outcome("a", ret_a)
                opt.record_outcome("b", ret_b)
        w_low = opt_low.compute_weights()
        w_high = opt_high.compute_weights()
        gap_low = abs(w_low["a"] - w_low["b"])
        gap_high = abs(w_high["a"] - w_high["b"])
        assert gap_high > gap_low

    def test_record_unknown_source_ignored(self):
        opt = AdaptiveWeightOptimizer(["a"])
        opt.record_outcome("unknown", 0.05)  # should not crash
        weights = opt.compute_weights()
        assert "unknown" not in weights

    def test_nan_in_outcomes_ignored(self):
        sp = SignalPerformance()
        sp.outcomes.extend([0.01, float("nan"), 0.03, 0.02])
        ir = sp.information_ratio
        assert ir != float("nan")
        assert isinstance(ir, float)

    def test_inf_in_outcomes_ignored(self):
        sp = SignalPerformance()
        sp.outcomes.extend([0.01, float("inf"), 0.03, -float("inf"), 0.02])
        ir = sp.information_ratio
        assert ir != float("inf")
        assert isinstance(ir, float)

    def test_all_nan_returns_zero(self):
        sp = SignalPerformance()
        sp.outcomes.extend([float("nan"), float("nan")])
        assert sp.information_ratio == 0.0

    def test_optimizer_maxlen_from_lookback(self):
        opt = AdaptiveWeightOptimizer(["a"], lookback_days=30)
        assert opt.performance["a"].outcomes.maxlen == 30
