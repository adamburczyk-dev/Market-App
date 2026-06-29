"""Concrete walk-forward revalidation backed by the momentum backtest engine.

Wires the orphaned ``ContinuousWalkForward`` (abstract) to a real backtest:
``_run_backtest`` measures the OOS Sharpe over the trailing ``oos_window_days``
bars (warmed up by the in-sample history before them). The base class compares
that against the strategy's original OOS Sharpe and recommends a status.
"""

import structlog

from src.core.continuous_validation import ContinuousWalkForward
from src.core.engine import BacktestParams, run_backtest

logger = structlog.get_logger()


class EngineWalkForward(ContinuousWalkForward):
    def __init__(
        self,
        params: BacktestParams | None = None,
        oos_window_days: int = 126,
        is_window_days: int = 252,
        degradation_threshold: float = 0.40,
    ) -> None:
        super().__init__(oos_window_days, is_window_days, degradation_threshold)
        self._params = params or BacktestParams()

    def _merge_params(self, strategy_params: dict) -> BacktestParams:
        """Overlay caller-supplied params on the defaults (unknown keys ignored)."""
        base = self._params
        return BacktestParams(
            lookback=int(strategy_params.get("lookback", base.lookback)),
            entry_momentum=float(strategy_params.get("entry_momentum", base.entry_momentum)),
            cost_bps=float(strategy_params.get("cost_bps", base.cost_bps)),
        )

    async def _run_backtest(
        self,
        strategy_name: str,
        strategy_params: dict,
        ohlcv_data: list[dict],
    ) -> float:
        closes = [float(bar["close"]) for bar in ohlcv_data]
        params = self._merge_params(strategy_params)
        # OOS = trailing oos_window_days bars; earlier bars warm up the lookback.
        oos_start = max(len(closes) - self.oos_window_days, 0)
        result = run_backtest(closes, params, start_index=oos_start)
        logger.info(
            "Walk-forward OOS backtest",
            strategy=strategy_name,
            oos_sharpe=round(result.sharpe_ratio, 4),
            oos_bars=result.n_bars,
            n_trades=result.n_trades,
        )
        return result.sharpe_ratio
