"""Strategy decay monitoring — detect degrading strategies early."""

from dataclasses import dataclass
from datetime import date


@dataclass
class StrategyHealth:
    """Snapshot of strategy health metrics."""

    strategy_name: str
    check_date: date
    sharpe_30d: float
    sharpe_90d: float
    sharpe_180d: float
    win_rate_30d: float
    profit_factor_30d: float
    excess_return_vs_spy_30d: float
    status: str  # "active", "probation", "deactivated"
    reason: str


class StrategyDecayMonitor:
    """
    Monitors strategy performance and assigns health status.

    Three states:
    - ACTIVE: Sharpe >= 0.5, PF >= 1.2, WR >= 0.4
    - DEACTIVATED: Sharpe < 0, PF < 0.8, or probation > 30 days
    - PROBATION: everything else (between active and deactivated thresholds)

    Research: Harvey et al. (2016) "… and the Cross-Section of Expected Returns"
    — strategies decay; continuous monitoring is essential.
    """

    # Active thresholds
    ACTIVE_SHARPE_MIN = 0.5
    ACTIVE_PF_MIN = 1.2
    ACTIVE_WR_MIN = 0.4

    # Deactivation thresholds
    DEACTIVATE_SHARPE_MAX = 0.0
    DEACTIVATE_PF_MAX = 0.8
    PROBATION_MAX_DAYS = 30

    def evaluate(
        self,
        strategy_name: str,
        sharpe_30d: float,
        sharpe_90d: float,
        sharpe_180d: float,
        win_rate_30d: float,
        profit_factor_30d: float,
        excess_return_vs_spy_30d: float,
        days_in_probation: int = 0,
    ) -> StrategyHealth:
        """Evaluate strategy health and return status with reason."""
        today = date.today()

        # Check deactivation triggers first (most severe)
        if sharpe_30d < self.DEACTIVATE_SHARPE_MAX:
            return StrategyHealth(
                strategy_name=strategy_name,
                check_date=today,
                sharpe_30d=sharpe_30d,
                sharpe_90d=sharpe_90d,
                sharpe_180d=sharpe_180d,
                win_rate_30d=win_rate_30d,
                profit_factor_30d=profit_factor_30d,
                excess_return_vs_spy_30d=excess_return_vs_spy_30d,
                status="deactivated",
                reason="negative_sharpe",
            )

        if profit_factor_30d < self.DEACTIVATE_PF_MAX:
            return StrategyHealth(
                strategy_name=strategy_name,
                check_date=today,
                sharpe_30d=sharpe_30d,
                sharpe_90d=sharpe_90d,
                sharpe_180d=sharpe_180d,
                win_rate_30d=win_rate_30d,
                profit_factor_30d=profit_factor_30d,
                excess_return_vs_spy_30d=excess_return_vs_spy_30d,
                status="deactivated",
                reason="low_profit_factor",
            )

        if days_in_probation > self.PROBATION_MAX_DAYS:
            return StrategyHealth(
                strategy_name=strategy_name,
                check_date=today,
                sharpe_30d=sharpe_30d,
                sharpe_90d=sharpe_90d,
                sharpe_180d=sharpe_180d,
                win_rate_30d=win_rate_30d,
                profit_factor_30d=profit_factor_30d,
                excess_return_vs_spy_30d=excess_return_vs_spy_30d,
                status="deactivated",
                reason="probation_timeout",
            )

        # Check if all active thresholds met
        if (
            sharpe_30d >= self.ACTIVE_SHARPE_MIN
            and profit_factor_30d >= self.ACTIVE_PF_MIN
            and win_rate_30d >= self.ACTIVE_WR_MIN
        ):
            return StrategyHealth(
                strategy_name=strategy_name,
                check_date=today,
                sharpe_30d=sharpe_30d,
                sharpe_90d=sharpe_90d,
                sharpe_180d=sharpe_180d,
                win_rate_30d=win_rate_30d,
                profit_factor_30d=profit_factor_30d,
                excess_return_vs_spy_30d=excess_return_vs_spy_30d,
                status="active",
                reason="all_metrics_healthy",
            )

        # Otherwise: probation
        reasons = []
        if sharpe_30d < self.ACTIVE_SHARPE_MIN:
            reasons.append("low_sharpe")
        if profit_factor_30d < self.ACTIVE_PF_MIN:
            reasons.append("low_pf")
        if win_rate_30d < self.ACTIVE_WR_MIN:
            reasons.append("low_win_rate")

        return StrategyHealth(
            strategy_name=strategy_name,
            check_date=today,
            sharpe_30d=sharpe_30d,
            sharpe_90d=sharpe_90d,
            sharpe_180d=sharpe_180d,
            win_rate_30d=win_rate_30d,
            profit_factor_30d=profit_factor_30d,
            excess_return_vs_spy_30d=excess_return_vs_spy_30d,
            status="probation",
            reason=",".join(reasons),
        )
