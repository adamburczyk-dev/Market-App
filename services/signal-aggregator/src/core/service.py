"""SignalAggregatorService — weight, combine, cost-gate, and publish signals.

The event-driven path makes this service the decision node of the trading loop:
strategy signals land in a per-symbol buffer (with their order-driving context —
price/SL/TP), the macro regime contributes a market-wide directional bias, and
each update publishes a ``SignalAggregatedEvent`` that risk-mgmt sizes into
orders. Buffered strategy signals expire after ``signal_ttl_s`` (strategy is
silent on HOLD, so without a TTL a stale BUY/SELL would resurface on every
regime change).
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from trading_common.cost_filter import CostAwareFilter
from trading_common.events import (
    RegimeChangedEvent,
    SignalAggregatedEvent,
    SignalGeneratedEvent,
)

from src.core.adaptive_weights import AdaptiveWeightOptimizer
from src.core.aggregator import (
    AggregationResult,
    SignalComponent,
    combine,
    regime_to_component,
)
from src.events.publisher import Publisher

logger = structlog.get_logger()


@dataclass
class BufferedSignal:
    """Latest strategy component for a symbol plus its order-driving context."""

    component: SignalComponent
    price: float | None
    stop_loss: float | None
    take_profit: float | None
    strategy_name: str | None
    at: datetime


class SignalAggregatorService:
    def __init__(
        self,
        optimizer: AdaptiveWeightOptimizer,
        cost_filter: CostAwareFilter,
        publisher: Publisher,
        buy_threshold: float = 0.2,
        base_edge_bps: float = 200.0,
        signal_ttl_s: float = 86_400.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._cost = cost_filter
        self._publisher = publisher
        self._buy_threshold = buy_threshold
        self._base_edge_bps = base_edge_bps
        self._ttl_s = signal_ttl_s
        self._clock = clock or (lambda: datetime.now(UTC))
        # live event state: latest strategy signal per symbol + market-wide macro bias
        self._buffer: dict[str, BufferedSignal] = {}
        self._macro: SignalComponent | None = None

    def weights(self) -> dict[str, float]:
        return self._optimizer.compute_weights()

    def record_outcome(self, source: str, daily_return: float) -> None:
        """Feed a realized per-source outcome to the adaptive weighting."""
        self._optimizer.record_outcome(source, daily_return)

    # --- live event handlers (NATS-driven) ---

    async def handle_signal_generated(self, data: bytes) -> None:
        """A strategy (rule-based) signal → buffer it and re-aggregate its symbol."""
        event = SignalGeneratedEvent.model_validate_json(data)
        self._buffer[event.symbol] = BufferedSignal(
            component=SignalComponent(
                source="strategy", signal=event.signal, confidence=event.confidence
            ),
            price=event.price,
            stop_loss=event.stop_loss,
            take_profit=event.take_profit,
            strategy_name=event.strategy_name,
            at=self._clock(),
        )
        await self.aggregate_symbol(event.symbol)

    async def handle_regime_changed(self, data: bytes) -> None:
        """A macro regime change → update the market-wide bias, re-aggregate all symbols."""
        event = RegimeChangedEvent.model_validate_json(data)
        self._macro = regime_to_component(event.new_regime)
        logger.info("Macro bias updated", regime=event.new_regime, bias=self._macro)
        for symbol in list(self._buffer):
            await self.aggregate_symbol(symbol)

    def _expired(self, entry: BufferedSignal) -> bool:
        return (self._clock() - entry.at).total_seconds() > self._ttl_s

    async def aggregate_symbol(self, symbol: str) -> AggregationResult | None:
        """Aggregate a symbol from its buffered strategy signal + the macro bias.

        The strategy component is required (the macro bias alone never emits a
        per-symbol signal); an expired entry is pruned and yields None.
        """
        entry = self._buffer.get(symbol)
        if entry is not None and self._expired(entry):
            logger.info("Buffered signal expired", symbol=symbol, age_limit_s=self._ttl_s)
            del self._buffer[symbol]
            entry = None
        if entry is None:
            return None

        components = [entry.component]
        if self._macro is not None:
            components.append(self._macro)
        return await self.aggregate(
            symbol,
            components,
            price=entry.price,
            stop_loss=entry.stop_loss,
            take_profit=entry.take_profit,
            strategy_name=entry.strategy_name,
            levels_direction=entry.component.signal,
        )

    def _weights_for(self, sources: list[str]) -> dict[str, float]:
        """Optimizer weights restricted to the present sources, renormalized.

        A source the optimizer doesn't track gets the equal-weight baseline so a
        newly-added component still contributes.
        """
        full = self._optimizer.compute_weights()
        n = len(self._optimizer.sources)
        default = 1.0 / n if n else 1.0
        raw = {s: full.get(s, default) for s in sources}
        total = sum(raw.values()) or 1.0
        return {s: w / total for s, w in raw.items()}

    async def aggregate(
        self,
        symbol: str,
        components: list[SignalComponent],
        expected_return_bps: float | None = None,
        market_cap_tier: str = "large",
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        strategy_name: str | None = None,
        levels_direction: str | None = None,
    ) -> AggregationResult:
        """Combine components → weighted signal → cost gate → publish aggregated signal.

        ``price``/``stop_loss``/``take_profit`` are attached to the published event
        only when the final decision is actionable and matches ``levels_direction``
        (the direction the levels were computed for) — a BUY stop makes no sense on
        a SELL decision. An actionable aggregate without levels is published anyway
        but risk-mgmt will block it (no order without stop_loss).
        """
        weights = self._weights_for([c.source for c in components])
        signal, confidence, score = combine(components, weights, self._buy_threshold)

        cost_filtered = False
        if signal in ("BUY", "SELL"):
            edge_bps = (
                expected_return_bps
                if expected_return_bps is not None
                else confidence * self._base_edge_bps
            )
            profitable, _ = self._cost.is_profitable_after_costs(
                edge_bps, market_cap_tier=market_cap_tier
            )
            if not profitable:
                signal = "HOLD"
                cost_filtered = True

        attach = signal in ("BUY", "SELL") and (
            levels_direction is None or levels_direction == signal
        )
        if signal in ("BUY", "SELL") and not attach:
            logger.warning(
                "Actionable aggregate without matching levels — downstream will block",
                symbol=symbol,
                final_signal=signal,
                levels_direction=levels_direction,
            )

        result = AggregationResult(
            symbol=symbol,
            final_signal=signal,
            confidence=confidence,
            score=score,
            components_count=len(components),
            weights=weights,
            cost_filtered=cost_filtered,
            price=price if attach else None,
            stop_loss=stop_loss if attach else None,
            take_profit=take_profit if attach else None,
            strategy_name=strategy_name if attach else None,
        )

        await self._publisher.publish(
            SignalAggregatedEvent(
                symbol=symbol,
                final_signal=result.final_signal,
                confidence=result.confidence,
                components_count=result.components_count,
                price=result.price,
                stop_loss=result.stop_loss,
                take_profit=result.take_profit,
                strategy_name=result.strategy_name,
            )
        )
        logger.info(
            "Signal aggregated",
            symbol=symbol,
            final_signal=result.final_signal,
            confidence=round(confidence, 4),
            components=len(components),
            cost_filtered=cost_filtered,
        )
        return result
