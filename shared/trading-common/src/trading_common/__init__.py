"""trading-common — shared contracts for the trading system."""

from trading_common.events import (
    BaseEvent,
    CircuitBreakerLevel,
    CircuitBreakerTriggeredEvent,
    EventType,
    MarketDataUpdatedEvent,
    ModelDriftDetectedEvent,
    ModelRetrainedEvent,
    SignalGeneratedEvent,
    StrategyStatusChangedEvent,
)
from trading_common.risk_envelope import RiskEnvelope, RiskLimits
from trading_common.schemas import (
    Interval,
    OHLCVBar,
    PortfolioMetrics,
    Signal,
    TradingSignal,
)

__all__ = [
    "Interval",
    "OHLCVBar",
    "Signal",
    "TradingSignal",
    "PortfolioMetrics",
    "RiskEnvelope",
    "RiskLimits",
    "EventType",
    "BaseEvent",
    "CircuitBreakerLevel",
    "CircuitBreakerTriggeredEvent",
    "MarketDataUpdatedEvent",
    "ModelDriftDetectedEvent",
    "ModelRetrainedEvent",
    "SignalGeneratedEvent",
    "StrategyStatusChangedEvent",
]
