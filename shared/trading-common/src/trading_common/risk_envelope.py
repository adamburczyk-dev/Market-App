"""Lightweight pre-trade risk check — Layer 1 defense, active from day 1."""

from dataclasses import dataclass

from trading_common.schemas import TradingSignal


@dataclass
class RiskLimits:
    """Minimum risk rules enforced before any signal is published."""

    max_position_pct: float = 0.05
    max_portfolio_exposure_pct: float = 0.80
    max_single_loss_pct: float = 0.02
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    max_correlated_positions: int = 3
    min_confidence: float = 0.55


class RiskEnvelope:
    """
    Pre-trade gate. Every signal-generating service MUST call check_signal()
    before publishing SignalGeneratedEvent.

    Check order (first failure wins):
    1. Portfolio drawdown
    2. Daily loss
    3. Signal confidence
    4. Portfolio exposure
    5. Risk per trade (requires stop_loss)
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def check_signal(
        self,
        signal: TradingSignal,
        portfolio_value: float,
        current_exposure_pct: float,
        current_drawdown_pct: float,
        daily_loss_pct: float,
        sector_positions: dict[str, int],
    ) -> tuple[bool, str]:
        """Returns (approved, reason). Reason is structured for parsing."""
        # 1. Drawdown hard stop
        if abs(current_drawdown_pct) >= self.limits.max_drawdown_pct:
            return False, (
                f"portfolio_drawdown_{abs(current_drawdown_pct):.1%}"
                f"_exceeds_{self.limits.max_drawdown_pct:.1%}"
            )

        # 2. Daily loss hard stop
        if abs(daily_loss_pct) >= self.limits.max_daily_loss_pct:
            return False, (
                f"daily_loss_{abs(daily_loss_pct):.1%}_exceeds_{self.limits.max_daily_loss_pct:.1%}"
            )

        # 3. Confidence threshold
        if signal.confidence < self.limits.min_confidence:
            return False, (
                f"confidence_{signal.confidence:.2f}_below_{self.limits.min_confidence:.2f}"
            )

        # 4. Exposure limit
        if current_exposure_pct >= self.limits.max_portfolio_exposure_pct:
            return False, (
                f"exposure_{current_exposure_pct:.1%}"
                f"_exceeds_{self.limits.max_portfolio_exposure_pct:.1%}"
            )

        # 5. Risk per trade (only when stop_loss is provided)
        if signal.stop_loss is not None and portfolio_value > 0:
            risk_per_share = abs(signal.price - signal.stop_loss)
            if risk_per_share > 0:
                max_risk_amount = portfolio_value * self.limits.max_single_loss_pct
                max_shares = max_risk_amount / risk_per_share
                position_value = max_shares * signal.price
                if position_value / portfolio_value > self.limits.max_position_pct:
                    return False, "position_size_exceeds_limit_after_risk_sizing"

        return True, "approved"
