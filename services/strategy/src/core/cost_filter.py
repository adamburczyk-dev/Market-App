"""Transaction cost-aware trade filtering."""

from dataclasses import dataclass, field


@dataclass
class TransactionCosts:
    """Single-leg transaction cost components in basis points."""

    spread_bps: float = 5.0
    slippage_bps: float = 5.0
    market_impact_bps: float = 2.0

    @property
    def total_roundtrip_bps(self) -> float:
        """Total cost for open + close = 2 * single leg."""
        return 2 * (self.spread_bps + self.slippage_bps + self.market_impact_bps)


# Market cap tier cost multipliers — smaller caps have higher friction
CAP_TIER_MULTIPLIERS: dict[str, float] = {
    "large": 1.0,
    "mid": 1.5,
    "small": 2.5,
    "micro": 5.0,
}


@dataclass
class CostAwareFilter:
    """
    Filters trades where expected edge doesn't justify transaction costs.

    Rule: expected_return must exceed min_edge_multiple × adjusted costs.
    Adjusted costs = roundtrip_bps × cap_tier_multiplier.

    Research: Novy-Marx & Velikov (2016) — transaction costs eliminate
    most anomaly profits in small/micro caps.
    """

    costs: TransactionCosts = field(default_factory=TransactionCosts)
    min_edge_multiple: float = 2.0

    def is_profitable_after_costs(
        self,
        expected_return_bps: float,
        holding_period_days: int = 1,
        market_cap_tier: str = "large",
    ) -> tuple[bool, dict]:
        """
        Check if trade edge justifies costs.

        Returns (is_profitable, details_dict).
        """
        multiplier = CAP_TIER_MULTIPLIERS.get(market_cap_tier, 1.0)
        adjusted_cost_bps = self.costs.total_roundtrip_bps * multiplier
        required_edge_bps = adjusted_cost_bps * self.min_edge_multiple

        is_profitable = expected_return_bps >= required_edge_bps

        details = {
            "adjusted_cost_bps": adjusted_cost_bps,
            "required_edge_bps": required_edge_bps,
            "expected_return_bps": expected_return_bps,
            "edge_to_cost_ratio": (
                expected_return_bps / adjusted_cost_bps if adjusted_cost_bps > 0 else float("inf")
            ),
            "market_cap_tier": market_cap_tier,
        }

        return is_profitable, details
