"""RiskMgmtService — size signals into orders; manage the circuit breaker."""

import structlog
from trading_common.events import (
    CircuitBreakerTriggeredEvent,
    OrderRequestedEvent,
    SignalGeneratedEvent,
)

from src.core.circuit_breaker import CircuitBreaker
from src.core.portfolio import PortfolioState
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
    ) -> None:
        self._publisher = publisher
        self._sizer = sizer
        self._breaker = breaker
        self._portfolio = portfolio

    @property
    def portfolio(self) -> PortfolioState:
        return self._portfolio

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    async def handle_signal_event(self, data: bytes) -> None:
        signal = SignalGeneratedEvent.model_validate_json(data)
        await self.process_signal(signal)

    async def process_signal(self, signal: SignalGeneratedEvent) -> OrderRequestedEvent | None:
        """Size a signal into a risk-approved order, or block it."""
        if self._breaker.is_tripped:
            logger.warning(
                "Circuit breaker tripped — order blocked",
                symbol=signal.symbol,
                level=self._breaker.level,
            )
            return None
        if signal.signal not in ("BUY", "SELL"):
            return None
        if signal.stop_loss is None:
            logger.warning("Signal without stop_loss — blocked", symbol=signal.symbol)
            return None

        shares, reason = self._sizer.size(signal.price, signal.stop_loss, self._portfolio)
        if shares <= 0:
            logger.info("Signal blocked by sizing/regime", symbol=signal.symbol, reason=reason)
            return None

        order = OrderRequestedEvent(
            symbol=signal.symbol,
            side=signal.signal,
            quantity=float(shares),
            price=signal.price,
            strategy_name=signal.strategy_name,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        await self._publisher.publish(order)
        logger.info("Order requested", symbol=signal.symbol, side=signal.signal, quantity=shares)
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
