"""StrategyService — turn FeaturesReadyEvent into a risk-checked SignalGeneratedEvent."""

from dataclasses import dataclass
from datetime import UTC, datetime

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

from src.core.feature_client import FeatureClient
from src.core.health import StrategyHealthTracker
from src.core.momentum import MomentumParams, generate_signal
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
        health: StrategyHealthTracker,
        risk_envelope: RiskEnvelope,
        cost_filter: CostAwareFilter,
        params: MomentumParams,
        portfolio: PortfolioSnapshot,
        strategy_name: str,
        stop_loss_pct: float = 0.05,
        take_profit_rr: float = 2.0,
        expected_edge_bps: float = 200.0,
        market_cap_tier: str = "large",
        portfolio_client: PortfolioClient | None = None,
    ) -> None:
        self._client = client
        self._publisher = publisher
        self._health = health
        self._risk = risk_envelope
        self._cost = cost_filter
        self._params = params
        self._portfolio = portfolio
        self._portfolio_client = portfolio_client
        self._name = strategy_name
        self._stop_pct = stop_loss_pct
        self._tp_rr = take_profit_rr
        self._edge_bps = expected_edge_bps
        self._cap_tier = market_cap_tier

    @property
    def name(self) -> str:
        return self._name

    @property
    def health(self) -> StrategyHealthTracker:
        return self._health

    async def handle_features_ready_event(self, data: bytes) -> None:
        event = FeaturesReadyEvent.model_validate_json(data)
        await self.evaluate_symbol(event.symbol, Interval(event.interval))

    async def handle_revalidated_event(self, data: bytes) -> None:
        event = StrategyRevalidatedEvent.model_validate_json(data)
        await self.apply_revalidation(event)

    async def apply_revalidation(
        self, event: StrategyRevalidatedEvent
    ) -> StrategyStatusChangedEvent | None:
        """Apply a backtest walk-forward recommendation to this strategy's status.

        Backtest only *recommends* — strategy owns the status (per the
        StrategyRevalidatedEvent contract). A recommendation for another
        strategy is ignored; an unknown recommended_status raises so the
        subscriber terminates the message as poison. Publishes
        StrategyStatusChangedEvent only on an actual transition.
        """
        if event.strategy_name != self._name:
            logger.info(
                "Revalidation for another strategy — ignored",
                target=event.strategy_name,
                own=self._name,
            )
            return None
        status = RECOMMENDED_TO_STATUS.get(event.recommended_status)
        if status is None:
            raise ValueError(f"unknown recommended_status: {event.recommended_status}")
        old_status = self._health.apply_status(status)
        if old_status is None:
            logger.info("Revalidation confirmed current status", status=status)
            return None
        changed = StrategyStatusChangedEvent(
            strategy_name=self._name,
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
            old=old_status,
            new=status,
            current_oos_sharpe=event.current_oos_sharpe,
        )
        return changed

    async def evaluate_symbol(self, symbol: str, interval: Interval) -> SignalGeneratedEvent | None:
        if not self._health.is_active:
            logger.info("Strategy not active, skipping", status=self._health.status)
            return None

        ranked = await self._client.get_ranked(symbol, interval)
        raw = await self._client.get_features(symbol, interval)
        if ranked is None or raw is None:
            return None

        momentum_rank = ranked.features.get("momentum_20")
        rsi = raw.features.get("rsi_14")
        price = raw.features.get("close")
        if momentum_rank is None or rsi is None or price is None:
            return None

        signal, confidence = generate_signal(momentum_rank, rsi, self._params)
        if signal == Signal.HOLD:
            return None

        stop_loss, take_profit = self._protective_levels(signal, price)
        trading_signal = TradingSignal(
            symbol=symbol,
            strategy=self._name,
            signal=signal,
            confidence=confidence,
            price=price,
            timestamp=datetime.now(UTC),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        portfolio = await self._current_portfolio()
        approved, reason = self._risk.check_signal(
            trading_signal,
            portfolio_value=portfolio.value,
            current_exposure_pct=portfolio.exposure_pct,
            current_drawdown_pct=portfolio.drawdown_pct,
            daily_loss_pct=portfolio.daily_loss_pct,
            sector_positions={},
        )
        if not approved:
            logger.info("Signal rejected by RiskEnvelope", symbol=symbol, reason=reason)
            return None

        expected_return_bps = self._edge_bps * confidence
        profitable, details = self._cost.is_profitable_after_costs(
            expected_return_bps, market_cap_tier=self._cap_tier
        )
        if not profitable:
            logger.info(
                "Signal filtered by cost",
                symbol=symbol,
                required_edge_bps=details["required_edge_bps"],
                expected_return_bps=expected_return_bps,
            )
            return None

        event = SignalGeneratedEvent(
            symbol=symbol,
            strategy_name=self._name,
            signal=signal.value,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={"momentum_rank": momentum_rank, "rsi": rsi, "risk": reason},
        )
        await self._publisher.publish(event)
        logger.info("Signal published", symbol=symbol, signal=signal.value, confidence=confidence)
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

    def _protective_levels(self, signal: Signal, price: float) -> tuple[float, float]:
        distance = price * self._stop_pct
        if signal == Signal.BUY:
            return price - distance, price + distance * self._tp_rr
        return price + distance, price - distance * self._tp_rr

    async def update_health(
        self,
        sharpe_30d: float,
        sharpe_90d: float,
        sharpe_180d: float,
        win_rate_30d: float,
        profit_factor_30d: float,
        excess_return_vs_spy_30d: float,
        days_in_probation: int = 0,
    ) -> StrategyStatusChangedEvent | None:
        """Re-evaluate decay health; publish StrategyStatusChangedEvent on a status change."""
        health, old_status = self._health.evaluate(
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
            strategy_name=self._name,
            old_status=old_status,
            new_status=health.status,
            reason=health.reason,
            sharpe_90d=health.sharpe_90d,
            profit_factor_30d=health.profit_factor_30d,
        )
        await self._publisher.publish(event)
        return event
