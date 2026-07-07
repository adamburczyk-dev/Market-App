"""RiskMgmtService — size aggregated signals into orders; manage the circuit breaker.

The NATS path consumes ``SignalAggregatedEvent`` (the signal-aggregator's decision
— strategy + macro-regime combined); the manual ``POST /signal`` route still accepts
a raw ``SignalGeneratedEvent`` for ops/testing. Both funnel into the same
risk-check → size → publish pipeline.
"""

import structlog
from trading_common.events import (
    CircuitBreakerTriggeredEvent,
    OrderRequestedEvent,
    RegimeChangedEvent,
    SignalAggregatedEvent,
    SignalGeneratedEvent,
)

from src.core.circuit_breaker import CircuitBreaker
from src.core.portfolio import PortfolioState
from src.core.repository import NullStateRepository, StateRepository
from src.core.sizing import PositionSizer
from src.events.publisher import Publisher

logger = structlog.get_logger()


class RiskMgmtService:
    def __init__(
        self,
        publisher: Publisher,
        sizer: PositionSizer,
        breaker: CircuitBreaker,
        portfolio: PortfolioState,
        repository: StateRepository | None = None,
    ) -> None:
        self._publisher = publisher
        self._sizer = sizer
        self._breaker = breaker
        self._portfolio = portfolio
        self._repository = repository or NullStateRepository()

    async def restore(self) -> None:
        """Load persisted portfolio state and re-derive the circuit-breaker level."""
        snapshot = await self._repository.load()
        if snapshot is None:
            return
        self._portfolio.update(
            value=snapshot.get("value"),
            exposure_pct=snapshot.get("exposure_pct"),
            drawdown_pct=snapshot.get("drawdown_pct"),
            daily_loss_pct=snapshot.get("daily_loss_pct"),
            regime=snapshot.get("regime"),
        )
        self._breaker.evaluate(self._portfolio.drawdown_pct, self._portfolio.daily_loss_pct)
        logger.info("Restored portfolio", level=self._breaker.level, **self._portfolio.as_dict())

    @property
    def portfolio(self) -> PortfolioState:
        return self._portfolio

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    # --- event handlers (NATS-driven) ---

    async def handle_aggregated_event(self, data: bytes) -> None:
        event = SignalAggregatedEvent.model_validate_json(data)
        await self.process_aggregated(event)

    async def handle_regime_changed_event(self, data: bytes) -> None:
        """Apply a macro regime change → drives regime-aware exposure caps."""
        event = RegimeChangedEvent.model_validate_json(data)
        await self.update_portfolio(regime=event.new_regime)
        logger.info("Applied regime change", old=event.old_regime, new=event.new_regime)

    # --- signal processing ---

    async def process_aggregated(self, event: SignalAggregatedEvent) -> OrderRequestedEvent | None:
        """Size the aggregator's decision into a risk-approved order, or block it."""
        if event.final_signal not in ("BUY", "SELL"):
            return None
        return await self._risk_check_and_order(
            symbol=event.symbol,
            side=event.final_signal,
            price=event.price,
            stop_loss=event.stop_loss,
            take_profit=event.take_profit,
            strategy_name=event.strategy_name or "aggregated",
            sector=event.sector,
        )

    async def process_signal(self, signal: SignalGeneratedEvent) -> OrderRequestedEvent | None:
        """Size a raw strategy signal (manual POST /signal path)."""
        if signal.signal not in ("BUY", "SELL"):
            return None
        return await self._risk_check_and_order(
            symbol=signal.symbol,
            side=signal.signal,
            price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy_name=signal.strategy_name,
        )

    async def _risk_check_and_order(
        self,
        *,
        symbol: str,
        side: str,
        price: float | None,
        stop_loss: float | None,
        take_profit: float | None,
        strategy_name: str,
        sector: str | None = None,
    ) -> OrderRequestedEvent | None:
        if self._breaker.is_tripped:
            logger.warning(
                "Circuit breaker tripped — order blocked",
                symbol=symbol,
                level=self._breaker.level,
            )
            return None
        if price is None or stop_loss is None:
            logger.warning("Signal without price/stop_loss — blocked", symbol=symbol, side=side)
            return None

        shares, reason = self._sizer.size(price, stop_loss, self._portfolio, sector=sector)
        if shares <= 0:
            logger.info("Signal blocked by sizing/regime", symbol=symbol, reason=reason)
            return None

        order = OrderRequestedEvent(
            symbol=symbol,
            side=side,
            quantity=float(shares),
            price=price,
            strategy_name=strategy_name,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        await self._publisher.publish(order)
        logger.info("Order requested", symbol=symbol, side=side, quantity=shares)
        return order

    async def update_portfolio(
        self,
        value: float | None = None,
        exposure_pct: float | None = None,
        drawdown_pct: float | None = None,
        daily_loss_pct: float | None = None,
        regime: str | None = None,
    ) -> CircuitBreakerTriggeredEvent | None:
        """Update portfolio state and re-arm the breaker; publish on a level change."""
        self._portfolio.update(value, exposure_pct, drawdown_pct, daily_loss_pct, regime)
        result = self._breaker.evaluate(
            self._portfolio.drawdown_pct, self._portfolio.daily_loss_pct
        )
        await self._repository.save(self._portfolio.as_dict())
        if result.changed and result.level is not None:
            event = CircuitBreakerTriggeredEvent(
                level=result.level,
                trigger_metric=result.trigger_metric,
                current_value=result.current_value,
                threshold_value=result.threshold,
                action_taken=result.action,
            )
            await self._publisher.publish(event)
            logger.warning("Circuit breaker triggered", level=result.level, action=result.action)
            return event
        return None
