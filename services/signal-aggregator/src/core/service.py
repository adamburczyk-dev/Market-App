"""SignalAggregatorService — weight, combine, cost-gate, and publish signals."""

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


class SignalAggregatorService:
    def __init__(
        self,
        optimizer: AdaptiveWeightOptimizer,
        cost_filter: CostAwareFilter,
        publisher: Publisher,
        buy_threshold: float = 0.2,
        base_edge_bps: float = 200.0,
    ) -> None:
        self._optimizer = optimizer
        self._cost = cost_filter
        self._publisher = publisher
        self._buy_threshold = buy_threshold
        self._base_edge_bps = base_edge_bps
        # live event buffers: latest per-symbol per-source component + market-wide macro bias
        self._buffer: dict[str, dict[str, SignalComponent]] = {}
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
        self._buffer.setdefault(event.symbol, {})["strategy"] = SignalComponent(
            source="strategy", signal=event.signal, confidence=event.confidence
        )
        await self.aggregate_symbol(event.symbol)

    async def handle_regime_changed(self, data: bytes) -> None:
        """A macro regime change → update the market-wide bias, re-aggregate all symbols."""
        event = RegimeChangedEvent.model_validate_json(data)
        self._macro = regime_to_component(event.new_regime)
        logger.info("Macro bias updated", regime=event.new_regime, bias=self._macro)
        for symbol in list(self._buffer):
            await self.aggregate_symbol(symbol)

    def _components_for(self, symbol: str) -> list[SignalComponent]:
        """Per-symbol buffered components plus the market-wide macro bias."""
        components = list(self._buffer.get(symbol, {}).values())
        if self._macro is not None:
            components.append(self._macro)
        return components

    async def aggregate_symbol(self, symbol: str) -> AggregationResult | None:
        """Aggregate a symbol from its buffered components; None if nothing buffered."""
        components = self._components_for(symbol)
        if not components:
            return None
        return await self.aggregate(symbol, components)

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
    ) -> AggregationResult:
        """Combine components → weighted signal → cost gate → publish aggregated signal."""
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

        result = AggregationResult(
            symbol=symbol,
            final_signal=signal,
            confidence=confidence,
            score=score,
            components_count=len(components),
            weights=weights,
            cost_filtered=cost_filtered,
        )

        await self._publisher.publish(
            SignalAggregatedEvent(
                symbol=symbol,
                final_signal=signal,
                confidence=confidence,
                components_count=len(components),
            )
        )
        logger.info(
            "Signal aggregated",
            symbol=symbol,
            final_signal=signal,
            confidence=round(confidence, 4),
            components=len(components),
            cost_filtered=cost_filtered,
        )
        return result
