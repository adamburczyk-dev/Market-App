"""Walk-forward revalidation of a REGISTERED rule.

``_run_backtest`` measures the OOS Sharpe over the trailing ``oos_window_days``
bars (warmed up by the in-sample history before them). The base class compares
that against the strategy's original OOS Sharpe and recommends a status.

The bars travel through the abstract ``ohlcv_data: list[dict]`` contract as full
serialized bars rather than bare closes: a rule needs highs, lows and volume
(ATR, Donchian, liquidity), and the previous lossy ``{"close": …}`` dict is part
of why this path could only ever evaluate a price-momentum proxy of whatever
strategy it was asked about.
"""

import structlog
from trading_common.prices import adjusted_closes
from trading_common.schemas import OHLCVBar
from trading_common.strategies import StrategyRule

from src.core.continuous_validation import ContinuousWalkForward
from src.core.rule_engine import run_rule_backtest

logger = structlog.get_logger()


class RuleWalkForward(ContinuousWalkForward):
    def __init__(
        self,
        rule: StrategyRule,
        cost_bps: float = 5.0,
        oos_window_days: int = 126,
        is_window_days: int = 252,
        degradation_threshold: float = 0.40,
    ) -> None:
        super().__init__(oos_window_days, is_window_days, degradation_threshold)
        self._rule = rule
        self._cost_bps = cost_bps

    async def _run_backtest(
        self,
        strategy_name: str,
        strategy_params: dict,
        ohlcv_data: list[dict],
    ) -> float:
        bars = [OHLCVBar.model_validate(row) for row in ohlcv_data]
        # OOS = trailing oos_window_days bars; earlier bars warm the rule up.
        oos_start = max(len(bars) - self.oos_window_days, 0)
        result = run_rule_backtest(
            bars,
            self._rule,
            adjusted_closes(bars),
            cost_bps=self._cost_bps,
            params=strategy_params,
            start_index=oos_start,
        )
        logger.info(
            "Walk-forward OOS backtest",
            strategy=strategy_name,
            oos_sharpe=round(result.sharpe_ratio, 4),
            oos_bars=result.n_bars,
            n_trades=result.n_trades,
        )
        return result.sharpe_ratio
