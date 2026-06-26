"""Lightweight pre-trade risk check — Layer 1 defense, active from day 1."""

from dataclasses import dataclass

from trading_common.schemas import Signal, TradingSignal


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

    def __post_init__(self) -> None:
        for field_name in [
            "max_position_pct",
            "max_portfolio_exposure_pct",
            "max_single_loss_pct",
            "max_daily_loss_pct",
            "max_drawdown_pct",
            "min_confidence",
        ]:
            val = getattr(self, field_name)
            if not (0 < val <= 1.0):
                raise ValueError(f"{field_name} must be in (0, 1.0], got {val}")
        if self.max_correlated_positions < 1:
            raise ValueError(
                f"max_correlated_positions must be >= 1, got {self.max_correlated_positions}"
            )


class RiskEnvelope:
    """
    Pre-trade gate. Every signal-generating service MUST call check_signal()
    before publishing SignalGeneratedEvent.

    This is a pure risk *gate* — it does NOT size positions. Position sizing
    (drawdown-adaptive, regime-aware) is risk-mgmt's job (`adaptive_sizing`).

    Check order (first failure wins):
    1. Portfolio drawdown
    2. Daily loss
    3. Signal confidence
    4. Portfolio exposure
    5. Sector correlation
    6. Order requires stop_loss (BUY/SELL; HOLD exempt)
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
        signal_sector: str | None = None,
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

        # 5. Sector correlation check
        if signal_sector is not None:
            sector_count = sector_positions.get(signal_sector, 0)
            if sector_count >= self.limits.max_correlated_positions:
                return False, (
                    f"sector_{signal_sector}_has_{sector_count}"
                    f"_positions_limit_{self.limits.max_correlated_positions}"
                )

        # 6. Orders must carry a stop_loss (HOLD places no order, so it is exempt).
        if signal.signal in (Signal.BUY, Signal.SELL) and signal.stop_loss is None:
            return False, "missing_stop_loss"

        return True, "approved"
