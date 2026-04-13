"""Testy event definitions — weryfikacja kontraktów NATS."""

from datetime import datetime

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


class TestBaseEvent:
    def make_event(self, **kwargs):
        defaults = {
            "event_type": EventType.MARKET_DATA_UPDATED,
            "source_service": "market-data",
        }
        return {**defaults, **kwargs}

    def test_event_id_auto_generated(self):
        e = BaseEvent(**self.make_event())
        assert e.event_id is not None
        assert len(e.event_id) > 0

    def test_two_events_have_different_ids(self):
        e1 = BaseEvent(**self.make_event())
        e2 = BaseEvent(**self.make_event())
        assert e1.event_id != e2.event_id

    def test_timestamp_auto_set(self):
        e = BaseEvent(**self.make_event())
        assert isinstance(e.timestamp, datetime)

    def test_correlation_id_optional(self):
        e = BaseEvent(**self.make_event())
        assert e.correlation_id is None

    def test_correlation_id_can_be_set(self):
        e = BaseEvent(**self.make_event(correlation_id="trace-123"))
        assert e.correlation_id == "trace-123"

    def test_subject_returns_event_type_value(self):
        e = MarketDataUpdatedEvent(symbol="AAPL", interval="1d", rows_count=100)
        assert e.subject() == "market_data.updated"


class TestMarketDataUpdatedEvent:
    def test_valid_event(self):
        e = MarketDataUpdatedEvent(symbol="AAPL", interval="1d", rows_count=250)
        assert e.symbol == "AAPL"
        assert e.rows_count == 250
        assert e.event_type == EventType.MARKET_DATA_UPDATED
        assert e.source_service == "market-data"

    def test_serialization_roundtrip(self):
        e = MarketDataUpdatedEvent(symbol="MSFT", interval="1h", rows_count=100)
        data = e.model_dump()
        assert data["symbol"] == "MSFT"
        assert data["event_type"] == "market_data.updated"


class TestSignalGeneratedEvent:
    def test_valid_signal_event(self):
        e = SignalGeneratedEvent(
            symbol="AAPL",
            strategy_name="sma_crossover",
            signal="BUY",
            confidence=0.85,
            price=153.5,
        )
        assert e.signal == "BUY"
        assert e.confidence == 0.85
        assert e.source_service == "strategy"

    def test_metadata_defaults_empty(self):
        e = SignalGeneratedEvent(
            symbol="AAPL",
            strategy_name="rsi",
            signal="SELL",
            confidence=0.7,
            price=150.0,
        )
        assert e.metadata == {}


class TestCircuitBreakerTriggeredEvent:
    def test_valid_event_yellow(self):
        e = CircuitBreakerTriggeredEvent(
            level=CircuitBreakerLevel.YELLOW,
            trigger_metric="drawdown",
            current_value=0.09,
            threshold_value=0.08,
            action_taken="reduce_exposure_50pct",
        )
        assert e.level == CircuitBreakerLevel.YELLOW
        assert e.event_type == EventType.CIRCUIT_BREAKER_TRIGGERED
        assert e.source_service == "risk-mgmt"

    def test_valid_event_red(self):
        e = CircuitBreakerTriggeredEvent(
            level=CircuitBreakerLevel.RED,
            trigger_metric="daily_loss",
            current_value=0.06,
            threshold_value=0.05,
            action_taken="flatten_all",
        )
        assert e.level == CircuitBreakerLevel.RED

    def test_valid_event_black(self):
        e = CircuitBreakerTriggeredEvent(
            level=CircuitBreakerLevel.BLACK,
            trigger_metric="anomaly",
            current_value=10.0,
            threshold_value=1.0,
            action_taken="halt_system",
        )
        assert e.level == CircuitBreakerLevel.BLACK

    def test_subject_is_risk_circuit_breaker(self):
        e = CircuitBreakerTriggeredEvent(
            level=CircuitBreakerLevel.RED,
            trigger_metric="drawdown",
            current_value=0.16,
            threshold_value=0.15,
            action_taken="flatten_all",
        )
        assert e.subject() == "risk.circuit_breaker"

    def test_serialization_roundtrip(self):
        e = CircuitBreakerTriggeredEvent(
            level=CircuitBreakerLevel.YELLOW,
            trigger_metric="var_breach",
            current_value=0.12,
            threshold_value=0.10,
            action_taken="reduce_exposure_50pct",
        )
        data = e.model_dump()
        restored = CircuitBreakerTriggeredEvent.model_validate(data)
        assert restored.level == e.level
        assert restored.trigger_metric == e.trigger_metric
        assert restored.current_value == e.current_value

    def test_all_circuit_breaker_levels(self):
        levels = list(CircuitBreakerLevel)
        assert len(levels) == 3
        assert set(levels) == {
            CircuitBreakerLevel.YELLOW,
            CircuitBreakerLevel.RED,
            CircuitBreakerLevel.BLACK,
        }


class TestModelDriftDetectedEvent:
    def test_valid_event(self):
        e = ModelDriftDetectedEvent(
            model_id="xgb_growth_tech_v3",
            drift_type="feature_drift",
            severity="warning",
            recommended_action="monitor",
        )
        assert e.model_id == "xgb_growth_tech_v3"
        assert e.event_type == EventType.MODEL_DRIFT_DETECTED
        assert e.source_service == "ml-pipeline"

    def test_subject(self):
        e = ModelDriftDetectedEvent(
            model_id="rf_v1",
            drift_type="prediction_drift",
            severity="critical",
            recommended_action="retrain",
        )
        assert e.subject() == "ml.drift_detected"

    def test_serialization_roundtrip(self):
        e = ModelDriftDetectedEvent(
            model_id="lstm_v2",
            drift_type="performance_decay",
            severity="critical",
            recommended_action="deactivate",
        )
        data = e.model_dump()
        restored = ModelDriftDetectedEvent.model_validate(data)
        assert restored.model_id == e.model_id
        assert restored.drift_type == e.drift_type


class TestModelRetrainedEvent:
    def test_valid_event(self):
        e = ModelRetrainedEvent(
            model_id="xgb_value_v4",
            old_sharpe=0.3,
            new_sharpe=1.1,
            retrain_reason="sharpe_decay",
        )
        assert e.old_sharpe == 0.3
        assert e.new_sharpe == 1.1
        assert e.event_type == EventType.MODEL_RETRAINED
        assert e.source_service == "ml-pipeline"

    def test_subject(self):
        e = ModelRetrainedEvent(
            model_id="rf_v2",
            old_sharpe=0.5,
            new_sharpe=0.8,
            retrain_reason="feature_drift",
        )
        assert e.subject() == "ml.model_retrained"

    def test_serialization_roundtrip(self):
        e = ModelRetrainedEvent(
            model_id="catboost_v1",
            old_sharpe=-0.2,
            new_sharpe=0.9,
            retrain_reason="auto_retrain",
        )
        data = e.model_dump()
        restored = ModelRetrainedEvent.model_validate(data)
        assert restored.old_sharpe == e.old_sharpe
        assert restored.new_sharpe == e.new_sharpe


class TestStrategyStatusChangedEvent:
    def test_valid_event(self):
        e = StrategyStatusChangedEvent(
            strategy_name="sma_crossover",
            old_status="active",
            new_status="probation",
            reason="below_active_thresholds",
            sharpe_90d=0.3,
            profit_factor_30d=1.0,
        )
        assert e.strategy_name == "sma_crossover"
        assert e.new_status == "probation"
        assert e.event_type == EventType.STRATEGY_STATUS_CHANGED
        assert e.source_service == "strategy"

    def test_subject(self):
        e = StrategyStatusChangedEvent(
            strategy_name="rsi_bb",
            old_status="probation",
            new_status="deactivated",
            reason="probation_timeout_30d",
            sharpe_90d=-0.1,
            profit_factor_30d=0.7,
        )
        assert e.subject() == "strategy.status_changed"

    def test_serialization_roundtrip(self):
        e = StrategyStatusChangedEvent(
            strategy_name="momentum_12_1",
            old_status="deactivated",
            new_status="active",
            reason="all_thresholds_met",
            sharpe_90d=1.2,
            profit_factor_30d=1.8,
        )
        data = e.model_dump()
        restored = StrategyStatusChangedEvent.model_validate(data)
        assert restored.strategy_name == e.strategy_name
        assert restored.sharpe_90d == e.sharpe_90d


class TestEventTypes:
    def test_all_event_types_have_dot_notation(self):
        for event_type in EventType:
            assert "." in event_type.value, f"{event_type} should use dot notation"

    def test_event_types_unique(self):
        values = [e.value for e in EventType]
        assert len(values) == len(set(values)), "EventType values must be unique"

    def test_new_event_types_exist(self):
        assert EventType.CIRCUIT_BREAKER_TRIGGERED == "risk.circuit_breaker"
        assert EventType.MODEL_DRIFT_DETECTED == "ml.drift_detected"
        assert EventType.MODEL_RETRAINED == "ml.model_retrained"
        assert EventType.STRATEGY_STATUS_CHANGED == "strategy.status_changed"

    def test_event_type_count(self):
        assert len(EventType) == 14
