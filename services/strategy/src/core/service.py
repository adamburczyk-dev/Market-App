"""StrategyService — turn FeaturesReadyEvent into a risk-checked SignalGeneratedEvent."""

from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from trading_common.events import (
    FeaturesReadyEvent,
    SignalGeneratedEvent,
    StrategyStatusChangedEvent,
)
from trading_common.risk_envelope import RiskEnvelope
from trading_common.schemas import Interval, Signal, TradingSignal

from src.core.cost_filter import CostAwareFilter
from src.core.feature_client import FeatureClient
from src.core.health import StrategyHealthTracker
from src.core.momentum import MomentumParams, generate_signal
from src.events.publisher import Publisher

logger = structlog.get_logger()

# RiskEnvelope step-7 rejects when the risk-budgeted size exceeds the position
# cap. That is a *sizing* concern (risk-mgmt sizes positions down) rather than a
# hard risk violation, so at signal-generation time we treat it as advisory.
_SIZING_REASON = "position_size_exceeds_limit_after_risk_sizing"


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
    ) -> None:
        self._client = client
        self._publisher = publisher
        self._health = health
        self._risk = risk_envelope
        self._cost = cost_filter
        self._params = params
        self._portfolio = portfolio
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

    async def evaluate_symbol(
        self, symbol: str, interval: Interval
    ) -> SignalGeneratedEvent | None:
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

        approved, reason = self._risk.check_signal(
            trading_signal,
            portfolio_value=self._portfolio.value,
            current_exposure_pct=self._portfolio.exposure_pct,
            current_drawdown_pct=self._portfolio.drawdown_pct,
            daily_loss_pct=self._portfolio.daily_loss_pct,
            sector_positions={},
        )
        if not approved and reason != _SIZING_REASON:
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
