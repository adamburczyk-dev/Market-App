"""BacktestService — run backtests and walk-forward revalidation, publish results."""

import uuid

import structlog
from trading_common.events import BacktestCompletedEvent, StrategyRevalidatedEvent
from trading_common.schemas import Interval

from src.core.continuous_validation import WalkForwardResult
from src.core.engine import BacktestParams, BacktestResult, run_backtest
from src.core.market_data_client import MarketDataClient
from src.core.walk_forward import EngineWalkForward
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

    def _merge_params(self, overrides: dict | None) -> BacktestParams:
        base = self._params
        o = overrides or {}
        return BacktestParams(
            lookback=int(o.get("lookback", base.lookback)),
            entry_momentum=float(o.get("entry_momentum", base.entry_momentum)),
            cost_bps=float(o.get("cost_bps", base.cost_bps)),
        )

    async def run_backtest(
        self,
        strategy_name: str,
        symbol: str,
        interval: Interval,
        limit: int = 500,
        params: dict | None = None,
    ) -> BacktestResult:
        """Fetch history, run the full-sample backtest, publish BacktestCompletedEvent."""
        bars = await self._market.get_ohlcv(symbol, interval, limit=limit)
        closes = [bar.close for bar in bars]
        result = run_backtest(closes, self._merge_params(params))

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
        """Walk-forward revalidation: compare current OOS Sharpe vs the original baseline."""
        bars = await self._market.get_ohlcv(symbol, interval, limit=limit)
        ohlcv = [{"close": bar.close} for bar in bars]

        wf = EngineWalkForward(
            params=self._merge_params(params),
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
