"""Mutable portfolio state.

In-memory placeholder until execution feeds real positions/P&L. Exposure,
drawdown, daily loss and regime drive sizing and the circuit breaker.
"""

from dataclasses import dataclass, field


@dataclass
class PortfolioState:
    value: float = 100_000.0
    exposure_pct: float = 0.0  # fraction of portfolio in equities
    drawdown_pct: float = 0.0  # current drawdown (0..1)
    daily_loss_pct: float = 0.0  # today's loss (0..1, positive = loss)
    regime: str = "expansion"
    sector_positions: dict[str, int] = field(default_factory=dict)

    def update(
        self,
        value: float | None = None,
        exposure_pct: float | None = None,
        drawdown_pct: float | None = None,
        daily_loss_pct: float | None = None,
        regime: str | None = None,
    ) -> None:
        if value is not None:
            self.value = value
        if exposure_pct is not None:
            self.exposure_pct = exposure_pct
        if drawdown_pct is not None:
            self.drawdown_pct = drawdown_pct
        if daily_loss_pct is not None:
            self.daily_loss_pct = daily_loss_pct
        if regime is not None:
            self.regime = regime

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "exposure_pct": self.exposure_pct,
            "drawdown_pct": self.drawdown_pct,
            "daily_loss_pct": self.daily_loss_pct,
            "regime": self.regime,
        }
