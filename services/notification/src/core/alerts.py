"""Map domain events to Alerts (severity + human-readable title/message).

These are the events that warrant a human-facing notification:
- risk.circuit_breaker          → critical (RED/BLACK) / warning (YELLOW)
- order.filled                  → info
- backtest.strategy_revalidated → warning (probation/deactivate) / info (active)
- ml.drift_detected             → severity carried on the event
"""

from trading_common.events import (
    CircuitBreakerLevel,
    CircuitBreakerTriggeredEvent,
    ModelDriftDetectedEvent,
    OrderFilledEvent,
    StrategyRevalidatedEvent,
)

from src.core.channels import Alert


def from_circuit_breaker(event: CircuitBreakerTriggeredEvent) -> Alert:
    severity = "warning" if event.level == CircuitBreakerLevel.YELLOW else "critical"
    return Alert(
        severity=severity,
        title=f"Circuit breaker {event.level.value.upper()}",
        message=(
            f"{event.trigger_metric}={event.current_value:.2%} "
            f"(threshold {event.threshold_value:.2%}) → {event.action_taken}"
        ),
        source=event.subject(),
        metadata={"level": event.level.value, "action": event.action_taken},
    )


def from_order_filled(event: OrderFilledEvent) -> Alert:
    return Alert(
        severity="info",
        title=f"Order filled: {event.symbol}",
        message=f"{event.filled_quantity:g} @ {event.filled_price:.2f} (order {event.order_id})",
        source=event.subject(),
        metadata={"symbol": event.symbol, "order_id": event.order_id},
    )


def from_strategy_revalidated(event: StrategyRevalidatedEvent) -> Alert:
    demoted = event.recommended_status in ("probation", "deactivate")
    return Alert(
        severity="warning" if demoted else "info",
        title=f"Strategy revalidated: {event.strategy_name} → {event.recommended_status}",
        message=(
            f"OOS Sharpe {event.current_oos_sharpe:.2f} vs {event.original_oos_sharpe:.2f} "
            f"(degradation {event.degradation_pct:.0%})"
        ),
        source=event.subject(),
        metadata={"strategy": event.strategy_name, "status": event.recommended_status},
    )


def from_model_drift(event: ModelDriftDetectedEvent) -> Alert:
    return Alert(
        severity=event.severity if event.severity in ("warning", "critical") else "warning",
        title=f"Model drift: {event.model_id} ({event.drift_type})",
        message=f"Recommended action: {event.recommended_action}",
        source=event.subject(),
        metadata={"model_id": event.model_id, "drift_type": event.drift_type},
    )
