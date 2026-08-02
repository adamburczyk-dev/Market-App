"""StrategyService — turn one FeaturesReadyEvent into one signal PER ACTIVE RULE.

Until this stage the service ran a single hard-wired rule, and everything
downstream was built for many: the aggregator combines components, the adaptive
weights learn per source, the decay monitor demotes a strategy that stops
working. With one rule those mechanisms had nothing to distinguish, so they were
quietly idle rather than visibly missing.

Each rule now gets its own health tracker and emits its own
`SignalGeneratedEvent` carrying its own `strategy_name` — that name is the
identity every per-strategy account downstream is keyed on.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from trading_common.cost_filter import CostAwareFilter
from trading_common.events import (
    FeaturesReadyEvent,
    SignalGeneratedEvent,
    StrategyRevalidatedEvent,
    StrategyStatusChangedEvent,
)
from trading_common.risk_envelope import RiskEnvelope
from trading_common.schemas import Interval, Signal, TradingSignal
from trading_common.strategies import RuleOutput, StrategyRule

from src.core.feature_client import FeatureClient
from src.core.health import StrategyHealthTracker
from src.core.portfolio_client import PortfolioClient
from src.events.publisher import Publisher

logger = structlog.get_logger()

# Backtest recommends in the imperative ("deactivate"); the tracker holds states.
RECOMMENDED_TO_STATUS = {
    "active": "active",
    "probation": "probation",
    "deactivate": "deactivated",
}


@dataclass
class PortfolioSnapshot:
    """Placeholder portfolio state until risk-mgmt provides the real one."""

    value: float = 100_000.0
    exposure_pct: float = 0.0
    drawdown_pct: float = 0.0
    daily_loss_pct: float = 0.0


class StrategyService:
    def __init__(
        self,
        client: FeatureClient,
        publisher: Publisher,
        rules: list[StrategyRule],
        risk_envelope: RiskEnvelope,
        cost_filter: CostAwareFilter,
        portfolio: PortfolioSnapshot,
        rule_params: dict[str, dict[str, float]] | None = None,
        fallback_stop_pct: float = 0.05,
        expected_edge_bps: float = 200.0,
        market_cap_tier: str = "large",
        portfolio_client: PortfolioClient | None = None,
    ) -> None:
        if not rules:
            raise ValueError("StrategyService needs at least one rule")
        self._client = client
        self._publisher = publisher
        self._rules = {rule.name: rule for rule in rules}
        self._health = {rule.name: StrategyHealthTracker(rule.name) for rule in rules}
        self._risk = risk_envelope
        self._cost = cost_filter
        self._portfolio = portfolio
        self._portfolio_client = portfolio_client
        self._params = rule_params or {}
        self._fallback_stop_pct = fallback_stop_pct
        self._edge_bps = expected_edge_bps
        self._cap_tier = market_cap_tier

    @property
    def names(self) -> list[str]:
        return sorted(self._rules)

    def health_of(self, name: str) -> StrategyHealthTracker:
        try:
            return self._health[name]
        except KeyError:
            raise KeyError(f"strategy not enabled here: {name} (running: {self.names})") from None

    def statuses(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "status": self._health[name].status,
                "required_features": sorted(self._rules[name].required_features),
                # Reported separately because it says something operational: a
                # rule with ranks needs a universe, so it cannot be validated by
                # a single-symbol backtest.
                "required_ranks": sorted(self._rules[name].required_ranks),
            }
            for name in self.names
        ]

    async def handle_features_ready_event(self, data: bytes) -> None:
        event = FeaturesReadyEvent.model_validate_json(data)
        await self.evaluate_symbol(event.symbol, Interval(event.interval))

    async def handle_revalidated_event(self, data: bytes) -> None:
        event = StrategyRevalidatedEvent.model_validate_json(data)
        await self.apply_revalidation(event)

    async def apply_revalidation(
        self, event: StrategyRevalidatedEvent
    ) -> StrategyStatusChangedEvent | None:
        """Apply a backtest walk-forward recommendation to ONE strategy's status.

        Backtest only *recommends* — strategy owns the status (per the
        StrategyRevalidatedEvent contract). A recommendation for a strategy this
        instance does not run is ignored; an unknown recommended_status raises so
        the subscriber terminates the message as poison. Publishes
        StrategyStatusChangedEvent only on an actual transition.
        """
        tracker = self._health.get(event.strategy_name)
        if tracker is None:
            logger.info(
                "Revalidation for a strategy we do not run — ignored",
                target=event.strategy_name,
                running=self.names,
            )
            return None
        status = RECOMMENDED_TO_STATUS.get(event.recommended_status)
        if status is None:
            raise ValueError(f"unknown recommended_status: {event.recommended_status}")
        old_status = tracker.apply_status(status)
        if old_status is None:
            logger.info(
                "Revalidation confirmed current status",
                strategy=event.strategy_name,
                status=status,
            )
            return None
        changed = StrategyStatusChangedEvent(
            strategy_name=event.strategy_name,
            old_status=old_status,
            new_status=status,
            reason=(
                f"backtest_revalidation:{event.recommended_status}"
                f"_degradation_{event.degradation_pct:.0%}"
            ),
            sharpe_90d=event.current_oos_sharpe,
        )
        await self._publisher.publish(changed)
        logger.warning(
            "Strategy status changed by revalidation",
            strategy=event.strategy_name,
            old=old_status,
            new=status,
            current_oos_sharpe=event.current_oos_sharpe,
        )
        return changed

    async def evaluate_symbol(self, symbol: str, interval: Interval) -> list[SignalGeneratedEvent]:
        """Run every ACTIVE rule against one symbol. Returns what was published.

        The feature vectors are fetched once and shared: the rules disagree
        about what to do with them, not about what they are, and re-querying
        per rule would let two rules in one session see two different snapshots.
        """
        active = [name for name in self.names if self._health[name].is_active]
        if not active:
            logger.info("No active strategy, skipping", symbol=symbol)
            return []

        ranked = await self._client.get_ranked(symbol, interval)
        raw = await self._client.get_features(symbol, interval)
        if ranked is None or raw is None:
            return []
        price = raw.features.get("close")
        if price is None or price <= 0:
            return []

        portfolio = await self._current_portfolio()
        published: list[SignalGeneratedEvent] = []
        for name in active:
            event = await self._evaluate_rule(
                self._rules[name],
                symbol=symbol,
                ranked=ranked.features,
                raw=raw.features,
                price=price,
                portfolio=portfolio,
            )
            if event is not None:
                published.append(event)
        return published

    async def _evaluate_rule(
        self,
        rule: StrategyRule,
        *,
        symbol: str,
        ranked: dict[str, float],
        raw: dict[str, float],
        price: float,
        portfolio: PortfolioSnapshot,
    ) -> SignalGeneratedEvent | None:
        decision = rule.generate(ranked, raw, self._params.get(rule.name))
        if decision.signal == Signal.HOLD:
            return None

        stop_loss, take_profit = self._protective_levels(decision, price, raw)
        trading_signal = TradingSignal(
            symbol=symbol,
            strategy=rule.name,
            signal=decision.signal,
            confidence=decision.confidence,
            price=price,
            timestamp=datetime.now(UTC),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        approved, reason = self._risk.check_signal(
            trading_signal,
            portfolio_value=portfolio.value,
            current_exposure_pct=portfolio.exposure_pct,
            current_drawdown_pct=portfolio.drawdown_pct,
            daily_loss_pct=portfolio.daily_loss_pct,
            sector_positions={},
        )
        if not approved:
            logger.info(
                "Signal rejected by RiskEnvelope", symbol=symbol, strategy=rule.name, reason=reason
            )
            return None

        expected_return_bps = self._edge_bps * decision.confidence
        profitable, details = self._cost.is_profitable_after_costs(
            expected_return_bps, market_cap_tier=self._cap_tier
        )
        if not profitable:
            logger.info(
                "Signal filtered by cost",
                symbol=symbol,
                strategy=rule.name,
                required_edge_bps=details["required_edge_bps"],
                expected_return_bps=expected_return_bps,
            )
            return None

        metadata: dict[str, Any] = dict(decision.metadata)
        metadata["rule"] = decision.reason
        metadata["risk"] = reason
        event = SignalGeneratedEvent(
            symbol=symbol,
            strategy_name=rule.name,
            signal=decision.signal.value,
            confidence=decision.confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
        )
        await self._publisher.publish(event)
        logger.info(
            "Signal published",
            symbol=symbol,
            strategy=rule.name,
            signal=decision.signal.value,
            confidence=decision.confidence,
        )
        return event

    async def _current_portfolio(self) -> PortfolioSnapshot:
        """Live portfolio from risk-mgmt; falls back to the static placeholder."""
        if self._portfolio_client is None:
            return self._portfolio
        data = await self._portfolio_client.get_portfolio()
        if data is None:
            return self._portfolio
        return PortfolioSnapshot(
            value=data.get("value", self._portfolio.value),
            exposure_pct=data.get("exposure_pct", 0.0),
            drawdown_pct=data.get("drawdown_pct", 0.0),
            daily_loss_pct=data.get("daily_loss_pct", 0.0),
        )

    def _protective_levels(
        self, decision: RuleOutput, price: float, raw: dict[str, float]
    ) -> tuple[float, float]:
        """Stop distance in ATR units, converted against the RAW execution price.

        `atr_pct_14` is a fraction, so it is scale-free: multiplying it by the
        raw close is valid even though ATR itself is computed on the adjusted
        series. A volatility-scaled stop is the point — a flat 5% is a wide stop
        on a utility and a tight one on a small-cap, so the same rule was taking
        two different risks depending on which name it fired for.

        Falls back to the configured flat percentage when ATR is absent (a
        symbol with under 15 sessions of history). Skipping the signal instead
        would be the wrong trade-off: a stop that is merely less well scaled
        still bounds the loss, and RiskEnvelope refuses anything without one.
        """
        atr_pct = raw.get("atr_pct_14")
        if atr_pct is not None and atr_pct > 0:
            stop_fraction = atr_pct * decision.stop_atr_mult
        else:
            stop_fraction = self._fallback_stop_pct
        distance = price * stop_fraction
        if decision.signal == Signal.BUY:
            return price - distance, price + distance * decision.take_profit_rr
        return price + distance, price - distance * decision.take_profit_rr

    async def update_health(
        self,
        strategy_name: str,
        sharpe_30d: float,
        sharpe_90d: float,
        sharpe_180d: float,
        win_rate_30d: float,
        profit_factor_30d: float,
        excess_return_vs_spy_30d: float,
        days_in_probation: int = 0,
    ) -> StrategyStatusChangedEvent | None:
        """Re-evaluate ONE strategy's decay health; publish on a status change."""
        tracker = self.health_of(strategy_name)
        health, old_status = tracker.evaluate(
            sharpe_30d,
            sharpe_90d,
            sharpe_180d,
            win_rate_30d,
            profit_factor_30d,
            excess_return_vs_spy_30d,
            days_in_probation,
        )
        if old_status is None:
            return None
        event = StrategyStatusChangedEvent(
            strategy_name=strategy_name,
            old_status=old_status,
            new_status=health.status,
            reason=health.reason,
            sharpe_90d=health.sharpe_90d,
            profit_factor_30d=health.profit_factor_30d,
        )
        await self._publisher.publish(event)
        return event
