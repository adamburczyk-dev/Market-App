"""SignalAggregatorService — weight, combine, cost-gate, and publish signals.

The event-driven path makes this service the decision node of the trading loop:
strategy signals land in a buffer keyed by (symbol, strategy) with their
order-driving context (price/SL/TP), the macro regime contributes a market-wide
directional bias, and each update publishes a ``SignalAggregatedEvent`` that
risk-mgmt sizes into orders. Buffered strategy signals expire after
``signal_ttl_s`` (strategy is silent on HOLD, so without a TTL a stale BUY/SELL
would resurface on every regime change).

**The buffer is keyed by the PAIR.** It used to be keyed by symbol alone, so a
second rule firing on the same name overwrote the first and the winner was
whichever NATS delivered last. Every strategy also entered as one source
``"strategy"``, which left `AdaptiveWeightOptimizer` — already generic over
source names — unable to tell the rules apart. With a single rule live both
defects were invisible, which is exactly why they survived.
"""

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from trading_common.cost_filter import CostAwareFilter
from trading_common.events import (
    MlSignalGeneratedEvent,
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
from src.core.company_client import CompanyClient
from src.events.publisher import Publisher

logger = structlog.get_logger()


#: Prefix that turns a strategy name into a weighting source. The prefix is
#: what keeps a rule called "ml" or "macro" from colliding with those sources.
STRATEGY_SOURCE_PREFIX = "strategy:"


def strategy_source(name: str) -> str:
    return f"{STRATEGY_SOURCE_PREFIX}{name}"


@dataclass
class BufferedSignal:
    """Latest component from ONE strategy for a symbol, plus its order context."""

    component: SignalComponent
    price: float | None
    stop_loss: float | None
    take_profit: float | None
    strategy_name: str | None
    at: datetime


def select_levels(entries: Iterable[BufferedSignal], direction: str) -> BufferedSignal | None:
    """Which strategy's price/SL/TP the order should use, once the vote is in.

    Only entries that AGREE with the final direction are eligible — a stop
    computed for a BUY is nonsense on a SELL. Among those the most confident
    wins, and ties break on the strategy NAME: without a deterministic rule the
    same set of inputs would produce different orders depending on the order
    NATS happened to deliver them in.
    """
    matching = [e for e in entries if e.component.signal == direction]
    if not matching:
        return None
    return sorted(matching, key=lambda e: (-e.component.confidence, e.strategy_name or ""))[0]


@dataclass(frozen=True)
class Decision:
    """The vote, before anything is known about which levels it will carry."""

    signal: str
    confidence: float
    score: float
    weights: dict[str, float]
    cost_filtered: bool


@dataclass
class BufferedMlSignal:
    """Latest ML vote for a symbol (no levels — ML cannot trade alone)."""

    component: SignalComponent
    model_id: str
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
        company_client: CompanyClient | None = None,
        join_window_s: float = 5.0,
    ) -> None:
        self._optimizer = optimizer
        self._cost = cost_filter
        self._publisher = publisher
        self._buy_threshold = buy_threshold
        self._base_edge_bps = base_edge_bps
        self._ttl_s = signal_ttl_s
        self._clock = clock or (lambda: datetime.now(UTC))
        self._company = company_client
        # live event state: latest signal per (symbol, strategy), ML vote per
        # symbol, and one market-wide macro bias
        self._buffer: dict[str, dict[str, BufferedSignal]] = {}
        self._ml_buffer: dict[str, BufferedMlSignal] = {}
        self._macro: SignalComponent | None = None
        # N2: `features.ready` fans out to strategy AND ml-pipeline in parallel.
        # The rule path is a comparison, the ML path is an inference, so strategy
        # always arrives first. Publishing on every component meant risk-mgmt
        # sized the strategy-only aggregate into an order, and then sized the
        # ML-informed one into a SECOND order — doubling the position, with the
        # ML vote never influencing anything. Wait a short window, decide once.
        self._join_window_s = join_window_s
        self._pending: dict[str, asyncio.Task[None]] = {}

    def weights(self) -> dict[str, float]:
        return self._optimizer.compute_weights()

    def record_outcome(self, source: str, daily_return: float) -> None:
        """Feed a realized per-source outcome to the adaptive weighting."""
        self._optimizer.record_outcome(source, daily_return)

    # --- live event handlers (NATS-driven) ---

    async def schedule_decision(self, symbol: str) -> None:
        """Coalesce the components of one decision, then aggregate once.

        `features.ready` fans out to strategy and ml-pipeline in parallel; the
        rule path is a comparison and the ML path an inference, so strategy
        always wins the race. Aggregating on arrival therefore published a
        strategy-only decision first and an ML-informed one moments later —
        two decisions where the domain has one. Waiting a short window lets the
        slower component join the same decision.

        Concurrent components collapse into the pending decision; a window of 0
        decides immediately (tests and the ops route).
        """
        if self._join_window_s <= 0:
            await self.aggregate_symbol(symbol)
            return
        if symbol in self._pending:
            return  # already scheduled — later components just enrich the buffer
        self._pending[symbol] = asyncio.create_task(self._decide_after_window(symbol))

    async def _decide_after_window(self, symbol: str) -> None:
        try:
            await asyncio.sleep(self._join_window_s)
            await self.aggregate_symbol(symbol)
        except asyncio.CancelledError:  # shutdown
            raise
        except Exception as exc:  # noqa: BLE001 - one symbol must not kill the loop
            logger.warning("Deferred aggregation failed", symbol=symbol, error=str(exc))
        finally:
            self._pending.pop(symbol, None)

    async def drain_pending(self) -> None:
        """Await every scheduled decision — used on shutdown and in tests."""
        tasks = list(self._pending.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_signal_generated(self, data: bytes) -> None:
        """A strategy (rule-based) signal → buffer it and re-aggregate its symbol.

        The buffer entry ages from the event's *emit* timestamp, not receive
        time — so a durable consumer replaying stream history (first start, or
        a rebuilt consumer) cannot resurrect stale signals past the TTL.
        """
        event = SignalGeneratedEvent.model_validate_json(data)
        self._buffer.setdefault(event.symbol, {})[event.strategy_name] = BufferedSignal(
            component=SignalComponent(
                source=strategy_source(event.strategy_name),
                signal=event.signal,
                confidence=event.confidence,
            ),
            price=event.price,
            stop_loss=event.stop_loss,
            take_profit=event.take_profit,
            strategy_name=event.strategy_name,
            at=event.timestamp,
        )
        await self.schedule_decision(event.symbol)

    async def handle_ml_signal(self, data: bytes) -> None:
        """An ML vote (plan §8, activates R11) → buffer it, re-aggregate its symbol.

        ML entries age from the event's emit timestamp like strategy signals;
        a symbol with only an ML vote never aggregates (strategy is required —
        ML modulates strategy-led decisions, it cannot trade alone).
        """
        event = MlSignalGeneratedEvent.model_validate_json(data)
        self._ml_buffer[event.symbol] = BufferedMlSignal(
            component=SignalComponent(
                source="ml", signal=event.signal, confidence=event.confidence
            ),
            model_id=event.model_id,
            at=event.timestamp,
        )
        await self.schedule_decision(event.symbol)

    async def handle_regime_changed(self, data: bytes) -> None:
        """A macro regime change → update the market-wide bias, re-aggregate all symbols."""
        event = RegimeChangedEvent.model_validate_json(data)
        self._macro = regime_to_component(event.new_regime)
        logger.info("Macro bias updated", regime=event.new_regime, bias=self._macro)
        for symbol in list(self._buffer):
            await self.schedule_decision(symbol)

    def _expired(self, entry: BufferedSignal | BufferedMlSignal) -> bool:
        return (self._clock() - entry.at).total_seconds() > self._ttl_s

    def _live_entries(self, symbol: str) -> list[BufferedSignal]:
        """Unexpired strategy entries for a symbol, pruning as it goes.

        Expiry is per ENTRY, not per symbol: one rule going quiet must not
        retire another rule's fresh signal, and a symbol whose entries all
        expired must not linger as an empty key.
        """
        by_strategy = self._buffer.get(symbol)
        if not by_strategy:
            return []
        for name in [n for n, entry in by_strategy.items() if self._expired(entry)]:
            logger.info(
                "Buffered signal expired", symbol=symbol, strategy=name, age_limit_s=self._ttl_s
            )
            del by_strategy[name]
        if not by_strategy:
            del self._buffer[symbol]
            return []
        # Sorted by strategy name so the component list — and therefore
        # `components_present` — does not depend on delivery order.
        return [by_strategy[name] for name in sorted(by_strategy)]

    async def aggregate_symbol(self, symbol: str) -> AggregationResult | None:
        """Aggregate a symbol from every buffered strategy + the ML vote + macro.

        At least one strategy component is required (neither the macro bias nor
        an ML vote alone emits a tradable per-symbol signal); expired entries
        are pruned — no live strategy signal yields None, an expired ML vote
        just drops out of the component list.
        """
        entries = self._live_entries(symbol)
        if not entries:
            return None

        components = [entry.component for entry in entries]
        ml_entry = self._ml_buffer.get(symbol)
        if ml_entry is not None and self._expired(ml_entry):
            logger.info("Buffered ML vote expired", symbol=symbol, model_id=ml_entry.model_id)
            del self._ml_buffer[symbol]
            ml_entry = None
        if ml_entry is not None:
            components.append(ml_entry.component)
        if self._macro is not None:
            components.append(self._macro)

        decision = self._decide(components)
        # Levels are chosen AFTER the vote, because which strategy's stop is
        # right depends on which direction won.
        chosen = (
            select_levels(entries, decision.signal) if decision.signal in ("BUY", "SELL") else None
        )
        sector = await self._company.get_sector(symbol) if self._company is not None else None
        return await self._publish(
            symbol,
            components,
            decision,
            price=chosen.price if chosen else None,
            stop_loss=chosen.stop_loss if chosen else None,
            take_profit=chosen.take_profit if chosen else None,
            strategy_name=chosen.strategy_name if chosen else None,
            levels_missing=decision.signal in ("BUY", "SELL") and chosen is None,
            sector=sector,
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

    def _decide(
        self,
        components: list[SignalComponent],
        expected_return_bps: float | None = None,
        market_cap_tier: str = "large",
    ) -> Decision:
        """Weighted combination + cost gate. No levels, no publishing."""
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
        return Decision(
            signal=signal,
            confidence=confidence,
            score=score,
            weights=weights,
            cost_filtered=cost_filtered,
        )

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
        sector: str | None = None,
    ) -> AggregationResult:
        """Combine explicitly-supplied components and publish (ops route + tests).

        ``price``/``stop_loss``/``take_profit`` are attached to the published event
        only when the final decision is actionable and matches ``levels_direction``
        (the direction the levels were computed for) — a BUY stop makes no sense on
        a SELL decision. The event-driven path (`aggregate_symbol`) instead PICKS
        the levels from whichever buffered strategy agrees with the outcome.
        """
        decision = self._decide(components, expected_return_bps, market_cap_tier)
        mismatch = decision.signal in ("BUY", "SELL") and not (
            levels_direction is None or levels_direction == decision.signal
        )
        return await self._publish(
            symbol,
            components,
            decision,
            price=None if mismatch else price,
            stop_loss=None if mismatch else stop_loss,
            take_profit=None if mismatch else take_profit,
            strategy_name=None if mismatch else strategy_name,
            levels_missing=mismatch,
            sector=sector,
        )

    async def _publish(
        self,
        symbol: str,
        components: list[SignalComponent],
        decision: Decision,
        *,
        price: float | None,
        stop_loss: float | None,
        take_profit: float | None,
        strategy_name: str | None,
        levels_missing: bool,
        sector: str | None,
    ) -> AggregationResult:
        """Build the result and emit SignalAggregatedEvent.

        An actionable aggregate with no usable levels is published anyway and
        risk-mgmt blocks it (no order without stop_loss) — but it is logged as a
        warning, because silently dropping it would hide a real disagreement
        between the vote and every strategy that produced levels.
        """
        if levels_missing:
            logger.warning(
                "Actionable aggregate without matching levels — downstream will block",
                symbol=symbol,
                final_signal=decision.signal,
            )

        actionable = decision.signal in ("BUY", "SELL")
        result = AggregationResult(
            symbol=symbol,
            final_signal=decision.signal,
            confidence=decision.confidence,
            score=decision.score,
            components_count=len(components),
            components_present=[c.source for c in components],
            weights=decision.weights,
            cost_filtered=decision.cost_filtered,
            price=price if actionable else None,
            stop_loss=stop_loss if actionable else None,
            take_profit=take_profit if actionable else None,
            strategy_name=strategy_name if actionable else None,
            sector=sector,
        )

        await self._publisher.publish(
            SignalAggregatedEvent(
                symbol=symbol,
                final_signal=result.final_signal,
                confidence=result.confidence,
                components_count=result.components_count,
                components_present=result.components_present,
                price=result.price,
                stop_loss=result.stop_loss,
                take_profit=result.take_profit,
                strategy_name=result.strategy_name,
                sector=result.sector,
            )
        )
        logger.info(
            "Signal aggregated",
            symbol=symbol,
            final_signal=result.final_signal,
            confidence=round(decision.confidence, 4),
            components=result.components_present,
            cost_filtered=decision.cost_filtered,
        )
        return result
