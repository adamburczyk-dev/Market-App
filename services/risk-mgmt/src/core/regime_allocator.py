"""Regime-aware equity exposure and sector allocation."""

from trading_common.sectors import normalize_sector


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
        "crisis": {"Consumer Staples", "Utilities", "Health Care"},
    }

    def max_exposure(self, regime: str) -> float:
        """Max fraction of portfolio in equity for given regime."""
        return self.MAX_EQUITY_EXPOSURE.get(regime, 0.60)

    def is_sector_allowed(self, regime: str, sector: str) -> bool:
        """True if sector is permitted in current regime.

        The incoming string is normalized to a GICS sector first (FLOW-8).
        Before that, matching was by exact (case-insensitive) text, so a profile
        saying "Technology" did not match the allow-list's "Information
        Technology" and the BUY was refused in a contraction — a data-entry
        difference acting as a risk decision. A string that normalizes to
        nothing is still refused: an unknown sector cannot be shown to be on
        the list, and this gate is conservative by design.
        """
        allowed = self.ALLOWED_SECTORS.get(regime)
        if allowed is None:
            return True
        canonical = normalize_sector(sector)
        return canonical is not None and canonical in allowed

    def required_cash_pct(self, regime: str) -> float:
        """Minimum cash allocation for given regime."""
        return 1.0 - self.max_exposure(regime)
