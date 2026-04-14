"""Tests for EarningsDecayCalculator — PEAD signal with exponential decay."""

import pytest

from src.core.calculators.earnings_decay import decay_weight, pead_signal, surprise_score


class TestDecayWeight:
    def test_day_zero_full_weight(self):
        assert decay_weight(0) == pytest.approx(1.0)

    def test_at_half_life(self):
        assert decay_weight(30) == pytest.approx(0.5)

    def test_double_half_life(self):
        assert decay_weight(60) == pytest.approx(0.25)

    def test_negative_days_returns_zero(self):
        assert decay_weight(-5) == 0.0

    def test_custom_half_life(self):
        assert decay_weight(10, half_life=10.0) == pytest.approx(0.5)

    def test_decay_is_monotonic(self):
        weights = [decay_weight(d) for d in range(0, 90, 10)]
        assert all(w1 >= w2 for w1, w2 in zip(weights, weights[1:], strict=False))

    def test_zero_half_life_raises(self):
        with pytest.raises(ValueError, match="half_life"):
            decay_weight(10, half_life=0.0)

    def test_negative_half_life_raises(self):
        with pytest.raises(ValueError, match="half_life"):
            decay_weight(10, half_life=-5.0)


class TestSurpriseScore:
    def test_positive_surprise(self):
        sue = surprise_score(actual_eps=2.0, consensus_eps=1.5, historical_std=0.25)
        assert sue == pytest.approx(2.0)

    def test_negative_surprise(self):
        sue = surprise_score(actual_eps=1.0, consensus_eps=1.5, historical_std=0.25)
        assert sue == pytest.approx(-2.0)

    def test_no_surprise(self):
        sue = surprise_score(actual_eps=1.5, consensus_eps=1.5, historical_std=0.25)
        assert sue == pytest.approx(0.0)

    def test_zero_std_returns_zero(self):
        sue = surprise_score(actual_eps=2.0, consensus_eps=1.5, historical_std=0.0)
        assert sue == 0.0

    def test_negative_std_returns_zero(self):
        assert surprise_score(2.0, 1.5, -0.1) == 0.0


class TestPeadSignal:
    def test_fresh_positive_surprise(self):
        signal = pead_signal(sue_score=2.0, days_since_earnings=0)
        assert signal == pytest.approx(2.0)

    def test_decayed_positive_surprise(self):
        signal = pead_signal(sue_score=2.0, days_since_earnings=30)
        assert signal == pytest.approx(1.0)  # half-life decay

    def test_negative_surprise_decays(self):
        signal = pead_signal(sue_score=-3.0, days_since_earnings=30)
        assert signal == pytest.approx(-1.5)

    def test_no_surprise_no_signal(self):
        signal = pead_signal(sue_score=0.0, days_since_earnings=10)
        assert signal == pytest.approx(0.0)
