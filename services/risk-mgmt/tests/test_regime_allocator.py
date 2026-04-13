"""Tests for RegimeAllocator — regime-aware equity exposure."""

import pytest

from src.core.regime_allocator import RegimeAllocator


class TestMaxExposure:
    def setup_method(self):
        self.alloc = RegimeAllocator()

    def test_expansion(self):
        assert self.alloc.max_exposure("expansion") == 0.90

    def test_recovery(self):
        assert self.alloc.max_exposure("recovery") == 0.80

    def test_slowdown(self):
        assert self.alloc.max_exposure("slowdown") == 0.60

    def test_contraction(self):
        assert self.alloc.max_exposure("contraction") == 0.35

    def test_crisis(self):
        assert self.alloc.max_exposure("crisis") == 0.15

    def test_unknown_defaults_to_60(self):
        assert self.alloc.max_exposure("unknown_regime") == 0.60


class TestIsSectorAllowed:
    def setup_method(self):
        self.alloc = RegimeAllocator()

    # Expansion: all allowed
    def test_expansion_allows_tech(self):
        assert self.alloc.is_sector_allowed("expansion", "Information Technology")

    def test_expansion_allows_energy(self):
        assert self.alloc.is_sector_allowed("expansion", "Energy")

    # Slowdown: selective
    def test_slowdown_allows_tech(self):
        assert self.alloc.is_sector_allowed("slowdown", "Information Technology")

    def test_slowdown_allows_staples(self):
        assert self.alloc.is_sector_allowed("slowdown", "Consumer Staples")

    def test_slowdown_allows_healthcare(self):
        assert self.alloc.is_sector_allowed("slowdown", "Health Care")

    def test_slowdown_allows_utilities(self):
        assert self.alloc.is_sector_allowed("slowdown", "Utilities")

    def test_slowdown_blocks_energy(self):
        assert not self.alloc.is_sector_allowed("slowdown", "Energy")

    def test_slowdown_blocks_financials(self):
        assert not self.alloc.is_sector_allowed("slowdown", "Financials")

    # Contraction: very defensive
    def test_contraction_allows_staples(self):
        assert self.alloc.is_sector_allowed("contraction", "Consumer Staples")

    def test_contraction_blocks_tech(self):
        assert not self.alloc.is_sector_allowed("contraction", "Information Technology")

    def test_contraction_blocks_energy(self):
        assert not self.alloc.is_sector_allowed("contraction", "Energy")

    # Crisis: minimal
    def test_crisis_allows_staples(self):
        assert self.alloc.is_sector_allowed("crisis", "Consumer Staples")

    def test_crisis_allows_utilities(self):
        assert self.alloc.is_sector_allowed("crisis", "Utilities")

    def test_crisis_blocks_healthcare(self):
        assert not self.alloc.is_sector_allowed("crisis", "Health Care")

    def test_crisis_blocks_tech(self):
        assert not self.alloc.is_sector_allowed("crisis", "Information Technology")

    # Unknown regime: all allowed (no entry in dict → None)
    def test_unknown_regime_allows_all(self):
        assert self.alloc.is_sector_allowed("unknown_regime", "Energy")

    # Recovery: all allowed
    def test_recovery_allows_all(self):
        assert self.alloc.is_sector_allowed("recovery", "Materials")


class TestRequiredCashPct:
    def setup_method(self):
        self.alloc = RegimeAllocator()

    def test_crisis_85pct_cash(self):
        assert self.alloc.required_cash_pct("crisis") == pytest.approx(0.85)

    def test_expansion_10pct_cash(self):
        assert self.alloc.required_cash_pct("expansion") == pytest.approx(0.10)

    def test_invariant_cash_plus_exposure_equals_1(self):
        """For all known regimes, cash + exposure must equal 1.0."""
        for regime in RegimeAllocator.MAX_EQUITY_EXPOSURE:
            cash = self.alloc.required_cash_pct(regime)
            exposure = self.alloc.max_exposure(regime)
            assert cash + exposure == pytest.approx(1.0), f"Invariant violated for {regime}"
