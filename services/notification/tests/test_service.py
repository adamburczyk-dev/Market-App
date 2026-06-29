"""Tests for NotificationService dispatch, filtering, and event handlers."""

import pytest
from trading_common.events import (
    CircuitBreakerLevel,
    CircuitBreakerTriggeredEvent,
    ModelDriftDetectedEvent,
    OrderFilledEvent,
    StrategyRevalidatedEvent,
)

from src.core.channels import Alert

from .conftest import CollectingChannel, FailingChannel, build_service


@pytest.mark.asyncio
async def test_dispatch_to_all_channels():
    a, b = CollectingChannel(), CollectingChannel()
    service = build_service([a, b])
    await service.dispatch(Alert("info", "t", "m", "manual"))
    assert len(a.sent) == 1
    assert len(b.sent) == 1


@pytest.mark.asyncio
async def test_min_severity_filters_below_threshold():
    ch = CollectingChannel()
    service = build_service([ch], min_severity="warning")
    await service.dispatch(Alert("info", "t", "m", "manual"))  # below → dropped
    await service.dispatch(Alert("critical", "t", "m", "manual"))  # above → sent
    assert len(ch.sent) == 1
    assert ch.sent[0].severity == "critical"


@pytest.mark.asyncio
async def test_channel_failure_isolated():
    good = CollectingChannel()
    service = build_service([FailingChannel(), good])
    # one channel raises; the other must still receive the alert, no exception bubbles
    await service.dispatch(Alert("critical", "t", "m", "manual"))
    assert len(good.sent) == 1


@pytest.mark.asyncio
async def test_recent_buffer_tracks_dispatched():
    ch = CollectingChannel()
    service = build_service([ch])
    for i in range(3):
        await service.dispatch(Alert("info", f"t{i}", "m", "manual"))
    recent = service.recent()
    assert [a.title for a in recent] == ["t0", "t1", "t2"]


@pytest.mark.asyncio
async def test_suppressed_alert_not_in_recent():
    service = build_service([CollectingChannel()], min_severity="critical")
    await service.dispatch(Alert("info", "t", "m", "manual"))
    assert service.recent() == []


@pytest.mark.asyncio
async def test_handle_circuit_breaker_event():
    ch = CollectingChannel()
    service = build_service([ch])
    e = CircuitBreakerTriggeredEvent(
        level=CircuitBreakerLevel.BLACK,
        trigger_metric="drawdown",
        current_value=0.16,
        threshold_value=0.15,
        action_taken="flatten_all",
    )
    await service.handle_circuit_breaker(e.model_dump_json().encode())
    assert ch.sent[0].severity == "critical"


@pytest.mark.asyncio
async def test_handle_order_filled_event():
    ch = CollectingChannel()
    service = build_service([ch])
    e = OrderFilledEvent(order_id="o1", symbol="MSFT", filled_quantity=10, filled_price=42.0)
    await service.handle_order_filled(e.model_dump_json().encode())
    assert ch.sent[0].severity == "info"
    assert "MSFT" in ch.sent[0].title


@pytest.mark.asyncio
async def test_handle_strategy_revalidated_event():
    ch = CollectingChannel()
    service = build_service([ch])
    e = StrategyRevalidatedEvent(
        strategy_name="s",
        original_oos_sharpe=1.0,
        current_oos_sharpe=-0.1,
        degradation_pct=1.1,
        recommended_status="deactivate",
        oos_window_days=126,
        is_window_days=252,
    )
    await service.handle_strategy_revalidated(e.model_dump_json().encode())
    assert ch.sent[0].severity == "warning"


@pytest.mark.asyncio
async def test_handle_model_drift_event():
    ch = CollectingChannel()
    service = build_service([ch])
    e = ModelDriftDetectedEvent(
        model_id="m1",
        drift_type="performance_decay",
        severity="critical",
        recommended_action="auto_retrain",
    )
    await service.handle_model_drift(e.model_dump_json().encode())
    assert ch.sent[0].severity == "critical"
