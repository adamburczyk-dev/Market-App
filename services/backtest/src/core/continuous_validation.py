"""Continuous walk-forward validation for strategy revalidation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WalkForwardResult:
    """Result of a walk-forward revalidation cycle."""

    strategy_name: str
    original_oos_sharpe: float
    current_oos_sharpe: float
    degradation_pct: float
    recommended_status: str  # "active", "probation", "deactivate"
    oos_window_days: int
    is_window_days: int


class ContinuousWalkForward(ABC):
    """
    Weekly walk-forward revalidation of live strategies.

    Compares current out-of-sample Sharpe against original.
    If degradation exceeds threshold → probation/deactivation.

    Research: White (2000) "Reality Check" — continuous OOS validation
    prevents overfitting to past market regimes.
    """

    def __init__(
        self,
        oos_window_days: int = 126,
        is_window_days: int = 252,
        degradation_threshold: float = 0.40,
    ) -> None:
        self.oos_window_days = oos_window_days
        self.is_window_days = is_window_days
        self.degradation_threshold = degradation_threshold

    @abstractmethod
    async def _run_backtest(
        self,
        strategy_name: str,
        strategy_params: dict,
        ohlcv_data: list[dict],
    ) -> float:
        """
        Run backtest and return OOS Sharpe ratio.

        Subclasses must implement actual backtest logic.
        """

    async def revalidate(
        self,
        strategy_name: str,
        original_oos_sharpe: float,
        ohlcv_data: list[dict],
        strategy_params: dict,
    ) -> WalkForwardResult:
        """
        Run walk-forward revalidation and recommend status.

        Returns WalkForwardResult with recommended_status:
        - "active": degradation within acceptable range
        - "probation": degradation exceeds threshold
        - "deactivate": current OOS Sharpe is negative
        """
        current_sharpe = await self._run_backtest(strategy_name, strategy_params, ohlcv_data)

        # Compute degradation
        if original_oos_sharpe != 0:
            degradation_pct = (original_oos_sharpe - current_sharpe) / abs(original_oos_sharpe)
        else:
            degradation_pct = 0.0

        # Determine status
        if current_sharpe < 0:
            status = "deactivate"
        elif degradation_pct >= self.degradation_threshold:
            status = "probation"
        else:
            status = "active"

        return WalkForwardResult(
            strategy_name=strategy_name,
            original_oos_sharpe=original_oos_sharpe,
            current_oos_sharpe=current_sharpe,
            degradation_pct=degradation_pct,
            recommended_status=status,
            oos_window_days=self.oos_window_days,
            is_window_days=self.is_window_days,
        )
