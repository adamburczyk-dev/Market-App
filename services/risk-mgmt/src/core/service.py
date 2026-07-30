"""RiskMgmtService — size aggregated signals into orders; manage the circuit breaker.

The NATS path consumes ``SignalAggregatedEvent`` (the signal-aggregator's decision
— strategy + macro-regime combined); the manual ``POST /signal`` route still accepts
a raw ``SignalGeneratedEvent`` for ops/testing. Both funnel into the same
risk-check → size → publish pipeline.
"""

import structlog
from trading_common.events import (
    CircuitBreakerTriggeredEvent,
    OrderIntent,
    OrderRequestedEvent,
    RegimeChangedEvent,
    SignalAggregatedEvent,
    SignalGeneratedEvent,
)

from src.core.circuit_breaker import CircuitBreaker, ResetResult
from src.core.order_ledger import OrderLedger, session_of
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
        ledger: OrderLedger | None = None,
    ) -> None:
        self._publisher = publisher
        self._sizer = sizer
        self._breaker = breaker
        self._portfolio = portfolio
        self._repository = repository or NullStateRepository()
        self._ledger = ledger or OrderLedger()

    async def restore(self) -> None:
        """Load persisted portfolio state, the order ledger AND the breaker latch.

        The latch has to be restored before re-evaluating, not derived from the
        metrics: a BLACK is held open until a human clears it, so if a restart
        re-derived the level from a recovered drawdown, restarting the container
        would silently BE the human reset — the easiest way to bypass the rule
        and one nobody would notice.
        """
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
        self._ledger.restore(snapshot.get("orders"))
        self._breaker.restore(snapshot.get("breaker"))
        self._breaker.evaluate(self._portfolio.drawdown_pct, self._portfolio.daily_loss_pct)
        logger.info(
            "Restored portfolio",
            level=self._breaker.level,
            latched=self._breaker.latched,
            **self._portfolio.as_dict(),
        )

    def _snapshot(self) -> dict:
        """Persisted state = portfolio + order ledger + breaker latch.

        The ledger keeps idempotency across a restart (otherwise a redelivered
        aggregate re-opens the same position); the latch keeps a human-restart
        requirement from being satisfied by a container restart.
        """
        return {
            **self._portfolio.as_dict(),
            "orders": self._ledger.snapshot(),
            "breaker": self._breaker.snapshot(),
        }

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
        """Size the aggregator's decision into a risk-approved order, or block it.

        Idempotent per (symbol, side, session): the aggregator re-publishes a
        decision whenever a component arrives (a late ML vote, a regime change),
        and re-sizing an already-acted-on BUY would double the position (N2).
        """
        if event.final_signal not in ("BUY", "SELL"):
            return None
        session = session_of(event.timestamp)
        if self._ledger.already_placed(event.symbol, event.final_signal, session):
            logger.info(
                "Aggregate already acted on this session — no duplicate order",
                symbol=event.symbol,
                side=event.final_signal,
                session=session,
                components=event.components_present,
            )
            return None
        order = await self._risk_check_and_order(
            symbol=event.symbol,
            side=event.final_signal,
            price=event.price,
            stop_loss=event.stop_loss,
            take_profit=event.take_profit,
            strategy_name=event.strategy_name or "aggregated",
            sector=event.sector,
        )
        if order is not None:
            # Only a placed order is recorded: a blocked one (halt, sizing, sector
            # cap) left no exposure, so a later component legitimately retries.
            self._ledger.record(event.symbol, event.final_signal, session)
            await self._repository.save(self._snapshot())
        return order

    async def process_signal(self, signal: SignalGeneratedEvent) -> OrderRequestedEvent | None:
        """Size a raw strategy signal (manual POST /signal path).

        Deliberately NOT ledger-gated: this route is ops/testing only, and an
        operator asking for an order twice means it twice.
        """
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
        intent: OrderIntent = OrderIntent.NEW,
    ) -> OrderRequestedEvent | None:
        # A halt stops NEW exposure. It must never stop a liquidation: BLACK's
        # answer to a >15% drawdown is to close positions, and closing is itself
        # an order — refusing it would be the opposite of what the breaker is
        # for, and the failure would only ever appear on the worst day.
        if self._breaker.is_tripped and intent is OrderIntent.NEW:
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
            intent=intent,
        )
        await self._publisher.publish(order)
        logger.info("Order requested", symbol=symbol, side=side, quantity=shares)
        return order

    async def reset_breaker(self) -> ResetResult:
        """Human clears a latched BLACK, and the cleared latch is persisted.

        Persisting matters as much as the clearing: the latch survives restarts
        by design, so a reset that lived only in memory would be undone by the
        next container restart and the operator would find the halt back.
        """
        result = self._breaker.reset(self._portfolio.drawdown_pct, self._portfolio.daily_loss_pct)
        if result.cleared:
            await self._repository.save(self._snapshot())
        return result

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
        await self._repository.save(self._snapshot())
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
