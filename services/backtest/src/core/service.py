"""BacktestService — run backtests and walk-forward revalidation, publish results."""

import uuid

import structlog
from trading_common.events import BacktestCompletedEvent, StrategyRevalidatedEvent
from trading_common.prices import adjusted_closes
from trading_common.schemas import Interval
from trading_common.strategies import get_strategy

from src.core.continuous_validation import WalkForwardResult
from src.core.engine import BacktestParams, BacktestResult
from src.core.market_data_client import MarketDataClient
from src.core.rule_engine import run_rule_backtest
from src.core.walk_forward import RuleWalkForward
from src.events.publisher import Publisher

logger = structlog.get_logger()


class BacktestService:
    def __init__(
        self,
        market_client: MarketDataClient,
        publisher: Publisher,
        default_params: BacktestParams | None = None,
        oos_window_days: int = 126,
        is_window_days: int = 252,
        degradation_threshold: float = 0.40,
    ) -> None:
        self._market = market_client
        self._publisher = publisher
        self._params = default_params or BacktestParams()
        self._oos_window_days = oos_window_days
        self._is_window_days = is_window_days
        self._degradation_threshold = degradation_threshold

    def _cost_bps(self, overrides: dict | None) -> float:
        """Execution cost for this run. Everything else in `params` is the
        RULE's, and is passed through untouched — the service has no business
        knowing which knobs a given rule has."""
        return float((overrides or {}).get("cost_bps", self._params.cost_bps))

    async def run_backtest(
        self,
        strategy_name: str,
        symbol: str,
        interval: Interval,
        limit: int = 500,
        params: dict | None = None,
    ) -> BacktestResult:
        """Fetch history, backtest THE NAMED RULE, publish BacktestCompletedEvent.

        `strategy_name` is resolved through the shared registry: an unknown name
        raises instead of being stamped on the built-in engine's numbers, which
        is what used to happen — any name produced the same result.

        Returns are measured on the ADJUSTED close, matching `revalidate` and
        the ML side. This path used to use the raw close, so two code paths in
        one service measured two different assets and only one of them counted
        dividends.
        """
        rule = get_strategy(strategy_name)
        bars = await self._market.get_ohlcv(symbol, interval, limit=limit)
        prices = adjusted_closes(bars)
        result = run_rule_backtest(
            bars, rule, prices, cost_bps=self._cost_bps(params), params=params
        )

        event = BacktestCompletedEvent(
            backtest_id=str(uuid.uuid4()),
            strategy_name=strategy_name,
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
        )
        await self._publisher.publish(event)
        logger.info(
            "Backtest completed",
            strategy=strategy_name,
            symbol=symbol,
            sharpe=round(result.sharpe_ratio, 4),
            total_return=round(result.total_return, 4),
            bars=result.n_bars,
        )
        return result

    async def revalidate(
        self,
        strategy_name: str,
        symbol: str,
        original_oos_sharpe: float,
        interval: Interval,
        limit: int = 500,
        params: dict | None = None,
    ) -> WalkForwardResult:
        """Walk-forward revalidation of THE NAMED RULE vs its original baseline.

        Returns are measured on the adjusted close — the same definition the ML
        side uses, otherwise a backtest and a model evaluation measure different
        assets.
        """
        rule = get_strategy(strategy_name)
        bars = await self._market.get_ohlcv(symbol, interval, limit=limit)
        # Whole bars, not just closes: the rules need highs/lows/volume.
        ohlcv = [bar.model_dump() for bar in bars]

        wf = RuleWalkForward(
            rule,
            cost_bps=self._cost_bps(params),
            oos_window_days=self._oos_window_days,
            is_window_days=self._is_window_days,
            degradation_threshold=self._degradation_threshold,
        )
        result = await wf.revalidate(strategy_name, original_oos_sharpe, ohlcv, params or {})

        event = StrategyRevalidatedEvent(
            strategy_name=result.strategy_name,
            original_oos_sharpe=result.original_oos_sharpe,
            current_oos_sharpe=result.current_oos_sharpe,
            degradation_pct=result.degradation_pct,
            recommended_status=result.recommended_status,
            oos_window_days=result.oos_window_days,
            is_window_days=result.is_window_days,
        )
        await self._publisher.publish(event)
        logger.info(
            "Strategy revalidated",
            strategy=strategy_name,
            recommended_status=result.recommended_status,
            current_oos_sharpe=round(result.current_oos_sharpe, 4),
            degradation_pct=round(result.degradation_pct, 4),
        )
        return result
