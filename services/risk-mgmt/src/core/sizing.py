"""Position sizing — drawdown-adaptive risk budget + regime exposure cap.

Wires the previously-orphaned DrawdownAdaptiveSizer and RegimeAllocator into the
runtime: this is where signals actually get sized (the RiskEnvelope only gates).
"""

from src.core.adaptive_sizing import DrawdownAdaptiveSizer
from src.core.portfolio import PortfolioState
from src.core.regime_allocator import RegimeAllocator


class PositionSizer:
    def __init__(
        self,
        base_risk_per_trade: float = 0.02,
        dd_scaling_start: float = 0.05,
        dd_scaling_end: float = 0.15,
        max_position_pct: float = 0.05,
    ) -> None:
        self._sizer = DrawdownAdaptiveSizer(
            base_risk_per_trade=base_risk_per_trade,
            dd_scaling_start=dd_scaling_start,
            dd_scaling_end=dd_scaling_end,
            max_position_pct=max_position_pct,
        )
        self._allocator = RegimeAllocator()

    def size(
        self,
        price: float,
        stop_loss: float,
        portfolio: PortfolioState,
        sector: str | None = None,
    ) -> tuple[int, str]:
        """Return (shares, reason). shares == 0 means the order is blocked."""
        max_exposure = self._allocator.max_exposure(portfolio.regime)
        if portfolio.exposure_pct >= max_exposure:
            return 0, f"regime_{portfolio.regime}_exposure_cap_{max_exposure:.0%}"

        if sector is not None and not self._allocator.is_sector_allowed(portfolio.regime, sector):
            return 0, f"regime_{portfolio.regime}_sector_{sector}_blocked"

        shares = self._sizer.position_size(
            portfolio.value, price, stop_loss, portfolio.drawdown_pct
        )
        if shares <= 0:
            return 0, "zero_size_after_drawdown_scaling"
        return shares, "sized"
