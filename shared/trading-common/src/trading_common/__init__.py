"""trading-common — shared contracts for the trading system."""

from trading_common.events import (
    AlertTriggeredEvent,
    BacktestCompletedEvent,
    BaseEvent,
    CircuitBreakerLevel,
    CircuitBreakerTriggeredEvent,
    EventType,
    MarketDataUpdatedEvent,
    ModelDriftDetectedEvent,
    ModelRetrainedEvent,
    ModelTrainedEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    OrderSubmittedEvent,
    RiskLimitBreachedEvent,
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
    "AlertTriggeredEvent",
    "BacktestCompletedEvent",
    "BaseEvent",
    "CircuitBreakerLevel",
    "CircuitBreakerTriggeredEvent",
    "EventType",
    "Interval",
    "MarketDataUpdatedEvent",
    "ModelDriftDetectedEvent",
    "ModelRetrainedEvent",
    "ModelTrainedEvent",
    "OHLCVBar",
    "OrderFilledEvent",
    "OrderRejectedEvent",
    "OrderSubmittedEvent",
    "PortfolioMetrics",
    "RiskEnvelope",
    "RiskLimits",
    "RiskLimitBreachedEvent",
    "Signal",
    "SignalGeneratedEvent",
    "StrategyStatusChangedEvent",
    "TradingSignal",
]
