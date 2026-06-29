"""Tests for event → Alert mapping (severity + content)."""

from trading_common.events import (
    CircuitBreakerLevel,
    CircuitBreakerTriggeredEvent,
    ModelDriftDetectedEvent,
    OrderFilledEvent,
    StrategyRevalidatedEvent,
)

from src.core import alerts


def test_circuit_breaker_red_is_critical():
    e = CircuitBreakerTriggeredEvent(
        level=CircuitBreakerLevel.RED,
        trigger_metric="daily_loss",
        current_value=0.06,
        threshold_value=0.05,
        action_taken="halt_trading",
    )
    alert = alerts.from_circuit_breaker(e)
    assert alert.severity == "critical"
    assert "RED" in alert.title
    assert alert.source == "risk.circuit_breaker"
    assert alert.metadata["action"] == "halt_trading"


def test_circuit_breaker_yellow_is_warning():
    e = CircuitBreakerTriggeredEvent(
        level=CircuitBreakerLevel.YELLOW,
        trigger_metric="drawdown",
        current_value=0.09,
        threshold_value=0.08,
        action_taken="warn",
    )
    assert alerts.from_circuit_breaker(e).severity == "warning"


def test_order_filled_is_info():
    e = OrderFilledEvent(order_id="o1", symbol="AAPL", filled_quantity=50, filled_price=100.0)
    alert = alerts.from_order_filled(e)
    assert alert.severity == "info"
    assert "AAPL" in alert.title
    assert alert.source == "order.filled"


def test_strategy_revalidated_probation_is_warning():
    e = StrategyRevalidatedEvent(
        strategy_name="momentum_rank",
        original_oos_sharpe=1.0,
        current_oos_sharpe=0.4,
        degradation_pct=0.6,
        recommended_status="probation",
        oos_window_days=126,
        is_window_days=252,
    )
    alert = alerts.from_strategy_revalidated(e)
    assert alert.severity == "warning"
    assert alert.metadata["status"] == "probation"


def test_strategy_revalidated_active_is_info():
    e = StrategyRevalidatedEvent(
        strategy_name="momentum_rank",
        original_oos_sharpe=1.0,
        current_oos_sharpe=1.1,
        degradation_pct=-0.1,
        recommended_status="active",
        oos_window_days=126,
        is_window_days=252,
    )
    assert alerts.from_strategy_revalidated(e).severity == "info"


def test_model_drift_severity_passthrough():
    e = ModelDriftDetectedEvent(
        model_id="m1",
        drift_type="feature_drift",
        severity="critical",
        recommended_action="auto_retrain",
    )
    alert = alerts.from_model_drift(e)
    assert alert.severity == "critical"
    assert "m1" in alert.title
    assert alert.source == "ml.drift_detected"
