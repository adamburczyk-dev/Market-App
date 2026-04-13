"""Regime-aware equity exposure and sector allocation."""


class RegimeAllocator:
    """
    Constrains max equity exposure and allowed sectors per market regime.

    Research: Ang & Bekaert (2004) — regime-conditional asset allocation
    improves Sharpe by 0.3-0.5 vs static allocation.
    """

    MAX_EQUITY_EXPOSURE: dict[str, float] = {
        "expansion": 0.90,
        "recovery": 0.80,
        "slowdown": 0.60,
        "contraction": 0.35,
        "crisis": 0.15,
    }

    ALLOWED_SECTORS: dict[str, set[str] | None] = {
        "expansion": None,  # All sectors allowed
        "recovery": None,
        "slowdown": {
            "Health Care",
            "Consumer Staples",
            "Utilities",
            "Information Technology",
        },
        "contraction": {"Consumer Staples", "Utilities", "Health Care"},
        "crisis": {"Consumer Staples", "Utilities"},
    }

    def max_exposure(self, regime: str) -> float:
        """Max fraction of portfolio in equity for given regime."""
        return self.MAX_EQUITY_EXPOSURE.get(regime, 0.60)

    def is_sector_allowed(self, regime: str, sector: str) -> bool:
        """True if sector is permitted in current regime."""
        allowed = self.ALLOWED_SECTORS.get(regime)
        if allowed is None:
            return True
        return sector in allowed

    def required_cash_pct(self, regime: str) -> float:
        """Minimum cash allocation for given regime."""
        return 1.0 - self.max_exposure(regime)
