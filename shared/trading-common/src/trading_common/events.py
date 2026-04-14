"""
Event definitions — kontrakt asynchronicznej komunikacji przez NATS.
Każdy event musi dziedziczyć z BaseEvent.
"""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(StrEnum):
    MARKET_DATA_UPDATED = "market_data.updated"
    FEATURES_COMPUTED = "features.computed"
    SIGNAL_GENERATED = "signal.generated"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_REJECTED = "order.rejected"
    RISK_LIMIT_BREACHED = "risk.limit_breached"
    CIRCUIT_BREAKER_TRIGGERED = "risk.circuit_breaker"
    MODEL_TRAINED = "ml.model_trained"
    MODEL_DRIFT_DETECTED = "ml.drift_detected"
    MODEL_RETRAINED = "ml.model_retrained"
    ALERT_TRIGGERED = "alert.triggered"
    BACKTEST_COMPLETED = "backtest.completed"
    STRATEGY_STATUS_CHANGED = "strategy.status_changed"


class CircuitBreakerLevel(StrEnum):
    YELLOW = "yellow"
    RED = "red"
    BLACK = "black"


class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_service: str
    correlation_id: str | None = None

    def subject(self) -> str:
        """NATS subject dla tego eventu."""
        return self.event_type.value


class MarketDataUpdatedEvent(BaseEvent):
    event_type: EventType = EventType.MARKET_DATA_UPDATED
    symbol: str
    interval: str
    rows_count: int
    source_service: str = "market-data"


class FeaturesComputedEvent(BaseEvent):
    event_type: EventType = EventType.FEATURES_COMPUTED
    symbol: str
    interval: str
    features_count: int
    source_service: str = "feature-engine"


class SignalGeneratedEvent(BaseEvent):
    event_type: EventType = EventType.SIGNAL_GENERATED
    symbol: str
    strategy_name: str
    signal: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    price: float
    metadata: dict = Field(default_factory=dict)
    source_service: str = "strategy"


class OrderSubmittedEvent(BaseEvent):
    event_type: EventType = EventType.ORDER_SUBMITTED
    order_id: str
    symbol: str
    side: str  # "BUY" | "SELL"
    quantity: float
    price: float
    source_service: str = "execution"


class OrderFilledEvent(BaseEvent):
    event_type: EventType = EventType.ORDER_FILLED
    order_id: str
    symbol: str
    filled_quantity: float
    filled_price: float
    source_service: str = "execution"


class RiskLimitBreachedEvent(BaseEvent):
    event_type: EventType = EventType.RISK_LIMIT_BREACHED
    symbol: str
    limit_type: str
    current_value: float
    limit_value: float
    source_service: str = "risk-mgmt"


class BacktestCompletedEvent(BaseEvent):
    event_type: EventType = EventType.BACKTEST_COMPLETED
    backtest_id: str
    strategy_name: str
    total_return: float
    sharpe_ratio: float | None = None
    source_service: str = "backtest"


class AlertTriggeredEvent(BaseEvent):
    event_type: EventType = EventType.ALERT_TRIGGERED
    alert_type: str
    message: str
    severity: str = "info"  # "info" | "warning" | "critical"
    source_service: str


class CircuitBreakerTriggeredEvent(BaseEvent):
    event_type: EventType = EventType.CIRCUIT_BREAKER_TRIGGERED
    level: CircuitBreakerLevel
    trigger_metric: str
    current_value: float
    threshold_value: float
    action_taken: str
    source_service: str = "risk-mgmt"


class ModelDriftDetectedEvent(BaseEvent):
    event_type: EventType = EventType.MODEL_DRIFT_DETECTED
    model_id: str
    drift_type: str
    severity: str  # "warning" | "critical"
    recommended_action: str
    source_service: str = "ml-pipeline"


class ModelRetrainedEvent(BaseEvent):
    event_type: EventType = EventType.MODEL_RETRAINED
    model_id: str
    old_sharpe: float
    new_sharpe: float
    retrain_reason: str
    source_service: str = "ml-pipeline"


class OrderRejectedEvent(BaseEvent):
    event_type: EventType = EventType.ORDER_REJECTED
    order_id: str
    symbol: str
    reason: str
    original_signal_id: str | None = None
    source_service: str = "execution"


class ModelTrainedEvent(BaseEvent):
    event_type: EventType = EventType.MODEL_TRAINED
    model_id: str
    model_type: str
    training_duration_s: float
    metrics: dict = Field(default_factory=dict)
    source_service: str = "ml-pipeline"


class StrategyStatusChangedEvent(BaseEvent):
    event_type: EventType = EventType.STRATEGY_STATUS_CHANGED
    strategy_name: str
    old_status: str
    new_status: str
    reason: str
    sharpe_90d: float
    profit_factor_30d: float
    source_service: str = "strategy"
