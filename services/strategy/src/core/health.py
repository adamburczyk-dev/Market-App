"""Strategy health tracking via StrategyDecayMonitor.

Gates signal output: only an ACTIVE strategy publishes signals. Health is
re-evaluated from performance metrics (fed by backtest/execution once they
exist); until then it stays ACTIVE.
"""

from src.core.decay_monitor import StrategyDecayMonitor, StrategyHealth


class StrategyHealthTracker:
    def __init__(self, strategy_name: str) -> None:
        self._name = strategy_name
        self._monitor = StrategyDecayMonitor()
        self._status = "active"
        self._health: StrategyHealth | None = None

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_active(self) -> bool:
        return self._status == "active"

    @property
    def health(self) -> StrategyHealth | None:
        return self._health

    def evaluate(
        self,
        sharpe_30d: float,
        sharpe_90d: float,
        sharpe_180d: float,
        win_rate_30d: float,
        profit_factor_30d: float,
        excess_return_vs_spy_30d: float,
        days_in_probation: int = 0,
    ) -> tuple[StrategyHealth, str | None]:
        """Re-evaluate health. Returns (health, old_status); old_status set only on change."""
        health = self._monitor.evaluate(
            self._name,
            sharpe_30d=sharpe_30d,
            sharpe_90d=sharpe_90d,
            sharpe_180d=sharpe_180d,
            win_rate_30d=win_rate_30d,
            profit_factor_30d=profit_factor_30d,
            excess_return_vs_spy_30d=excess_return_vs_spy_30d,
            days_in_probation=days_in_probation,
        )
        old = self._status
        self._status = health.status
        self._health = health
        return health, (old if old != health.status else None)
