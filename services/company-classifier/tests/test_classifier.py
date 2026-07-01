"""Tests for the rule-based company classifier."""

from src.core.classifier import (
    Style,
    ValuationMetrics,
    cap_tier,
    classify,
    route_model_stack,
)


class TestCapTier:
    def test_tiers(self):
        assert cap_tier(3e12) == "mega"
        assert cap_tier(50e9) == "large"
        assert cap_tier(5e9) == "mid"
        assert cap_tier(5e8) == "small"
        assert cap_tier(1e8) == "micro"
        assert cap_tier(None) == "unknown"


class TestModelRouting:
    def test_large_bucket(self):
        assert route_model_stack("growth", "mega") == "growth_largecap_v1"
        assert route_model_stack("value", "large") == "value_largecap_v1"

    def test_small_bucket(self):
        assert route_model_stack("growth", "mid") == "growth_smallcap_v1"
        assert route_model_stack("blend", "micro") == "blend_smallcap_v1"


class TestStyleFromMetrics:
    def test_growth(self):
        m = ValuationMetrics(
            pe_ratio=35, revenue_growth=0.25, earnings_growth=0.30, dividend_yield=0.0
        )
        r = classify("Information Technology", 3e12, m)
        assert r.style == Style.GROWTH
        assert r.basis == "metrics"
        assert r.growth_score > r.value_score

    def test_value(self):
        m = ValuationMetrics(pe_ratio=10, pb_ratio=1.1, dividend_yield=0.04)
        r = classify("Financials", 50e9, m)
        assert r.style == Style.VALUE
        assert r.value_score > r.growth_score

    def test_tie_is_blend(self):
        # one growth signal (rich P/E) and one value signal (high dividend) → tie → blend
        m = ValuationMetrics(pe_ratio=30, dividend_yield=0.05)
        r = classify("Industrials", 5e9, m)
        assert r.growth_score == r.value_score
        assert r.style == Style.BLEND


class TestStyleFallbacks:
    def test_sector_prior_growth(self):
        r = classify("Health Care", 8e9, None)
        assert r.style == Style.GROWTH
        assert r.basis == "sector"

    def test_sector_prior_value(self):
        r = classify("Utilities", 8e9, None)
        assert r.style == Style.VALUE
        assert r.basis == "sector"

    def test_unknown_sector_is_blend_default(self):
        r = classify("Conglomerates", 1e9, None)
        assert r.style == Style.BLEND
        assert r.basis == "default"

    def test_no_sector_no_metrics_is_blend(self):
        r = classify(None, None, None)
        assert r.style == Style.BLEND
        assert r.cap_tier == "unknown"


class TestModelStackReflectsClassification:
    def test_growth_smallcap_routing(self):
        m = ValuationMetrics(revenue_growth=0.4, earnings_growth=0.5)
        r = classify("Consumer Discretionary", 1.5e9, m)  # mid → small bucket
        assert r.style == Style.GROWTH
        assert r.model_stack == "growth_smallcap_v1"

    def test_metrics_override_sector_prior(self):
        # a "value" sector but clearly-growth metrics → metrics win
        m = ValuationMetrics(
            pe_ratio=40, revenue_growth=0.3, earnings_growth=0.3, dividend_yield=0.0
        )
        r = classify("Utilities", 50e9, m)
        assert r.style == Style.GROWTH
        assert r.basis == "metrics"
