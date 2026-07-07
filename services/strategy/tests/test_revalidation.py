"""Tests for R7 — strategy consumes backtest walk-forward revalidations.

Backtest recommends ("active" | "probation" | "deactivate"); strategy owns the
status: it applies the recommendation to its health tracker and publishes
StrategyStatusChangedEvent only on an actual transition.
"""

import pytest
from trading_common.events import EventType, StrategyRevalidatedEvent

from src.core.health import StrategyHealthTracker

from .conftest import build_service, buy_client


def revalidation(name: str = "momentum_rank", recommended: str = "probation", **kwargs):
    defaults = {
        "strategy_name": name,
        "original_oos_sharpe": 1.2,
        "current_oos_sharpe": 0.4,
        "degradation_pct": 0.67,
        "recommended_status": recommended,
        "oos_window_days": 126,
        "is_window_days": 252,
    }
    return StrategyRevalidatedEvent(**{**defaults, **kwargs})


# --- StrategyHealthTracker.apply_status ---


def test_apply_status_returns_old_on_change():
    tracker = StrategyHealthTracker("momentum_rank")
    assert tracker.apply_status("probation") == "active"
    assert tracker.status == "probation"


def test_apply_status_none_when_unchanged():
    tracker = StrategyHealthTracker("momentum_rank")
    assert tracker.apply_status("active") is None


def test_apply_status_rejects_unknown():
    tracker = StrategyHealthTracker("momentum_rank")
    with pytest.raises(ValueError, match="unknown strategy status"):
        tracker.apply_status("zombie")
    assert tracker.status == "active"  # unchanged


# --- StrategyService.apply_revalidation ---


@pytest.mark.asyncio
async def test_probation_recommendation_changes_status_and_publishes():
    from src.events.publisher import NullPublisher

    publisher = NullPublisher()
    service = build_service(buy_client(), publisher=publisher)
    changed = await service.apply_revalidation(revalidation(recommended="probation"))
    assert changed is not None
    assert changed.event_type == EventType.STRATEGY_STATUS_CHANGED
    assert changed.old_status == "active"
    assert changed.new_status == "probation"
    assert changed.sharpe_90d == 0.4  # current OOS sharpe carried into the audit event
    assert changed.profit_factor_30d is None  # revalidation has no PF to report
    assert service.health.status == "probation"
    assert publisher.published == [changed]


@pytest.mark.asyncio
async def test_deactivate_recommendation_suppresses_signals():
    service = build_service(buy_client())
    changed = await service.apply_revalidation(revalidation(recommended="deactivate"))
    assert changed is not None
    assert changed.new_status == "deactivated"  # imperative mapped to tracker state
    # a deactivated strategy stops emitting signals
    from trading_common.schemas import Interval

    assert await service.evaluate_symbol("AAPL", Interval.D1) is None


@pytest.mark.asyncio
async def test_active_recommendation_reactivates():
    service = build_service(buy_client())
    service.health.apply_status("probation")
    changed = await service.apply_revalidation(
        revalidation(recommended="active", current_oos_sharpe=1.1, degradation_pct=0.0)
    )
    assert changed is not None
    assert changed.old_status == "probation"
    assert changed.new_status == "active"
    assert service.health.is_active


@pytest.mark.asyncio
async def test_confirming_recommendation_publishes_nothing():
    from src.events.publisher import NullPublisher

    publisher = NullPublisher()
    service = build_service(buy_client(), publisher=publisher)
    assert await service.apply_revalidation(revalidation(recommended="active")) is None
    assert publisher.published == []


@pytest.mark.asyncio
async def test_other_strategys_revalidation_is_ignored():
    service = build_service(buy_client())
    result = await service.apply_revalidation(
        revalidation(name="sma_crossover", recommended="deactivate")
    )
    assert result is None
    assert service.health.is_active  # untouched


@pytest.mark.asyncio
async def test_unknown_recommendation_raises_for_poison_term():
    service = build_service(buy_client())
    with pytest.raises(ValueError, match="unknown recommended_status"):
        await service.apply_revalidation(revalidation(recommended="explode"))


@pytest.mark.asyncio
async def test_handle_revalidated_event_parses_bytes():
    service = build_service(buy_client())
    payload = revalidation(recommended="probation").model_dump_json().encode()
    await service.handle_revalidated_event(payload)
    assert service.health.status == "probation"
