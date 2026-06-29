"""Tests for MLPipelineService — drift orchestration + event publishing."""

import pytest
from trading_common.events import EventType

from src.events.publisher import NullPublisher

from .conftest import build_service, normal_samples


def stable_ref() -> dict[str, list[float]]:
    return {"mom": normal_samples(0, 1, seed=1), "rsi": normal_samples(50, 10, seed=2)}


@pytest.mark.asyncio
async def test_unknown_model_returns_none():
    service = build_service()
    report = await service.check_drift("ghost", {"mom": [1.0]}, 1.0, 1.0, 0.6)
    assert report is None


@pytest.mark.asyncio
async def test_stable_distribution_no_event():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    service.register_baseline("m1", stable_ref(), baseline_sharpe=1.0)
    current = {"mom": normal_samples(0, 1, seed=9), "rsi": normal_samples(50, 10, seed=10)}
    report = await service.check_drift("m1", current, 1.1, 1.0, 0.60)
    assert report is not None
    assert report.recommended_action == "no_action"
    assert publisher.published == []


@pytest.mark.asyncio
async def test_feature_drift_publishes_critical_event():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    service.register_baseline("m1", stable_ref(), baseline_sharpe=1.0)
    # big shift in mom → PSI well above 0.20
    current = {"mom": normal_samples(3, 1, seed=9), "rsi": normal_samples(50, 10, seed=10)}
    report = await service.check_drift("m1", current, 1.0, 1.0, 0.60)
    assert report.needs_retrain is True
    assert "mom" in report.features_drifted
    assert len(publisher.published) == 1
    event = publisher.published[0]
    assert event.event_type == EventType.MODEL_DRIFT_DETECTED
    assert event.drift_type == "feature_drift"
    assert event.severity == "critical"
    assert event.recommended_action == "auto_retrain"
    assert event.source_service == "ml-pipeline"


@pytest.mark.asyncio
async def test_sharpe_decay_publishes_performance_event():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    service.register_baseline("m1", stable_ref(), baseline_sharpe=1.0)
    stable_current = {"mom": normal_samples(0, 1, seed=9), "rsi": normal_samples(50, 10, seed=10)}
    # rolling 30d Sharpe collapses vs baseline 1.0 → decay -0.5 < -0.30
    report = await service.check_drift("m1", stable_current, 0.5, 0.8, 0.60)
    assert report.needs_retrain is True
    assert report.features_drifted == []
    event = publisher.published[0]
    assert event.drift_type == "performance_decay"
    assert event.severity == "critical"


@pytest.mark.asyncio
async def test_prediction_shift_only_is_warning():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    service.register_baseline(
        "m1", stable_ref(), baseline_sharpe=1.0, prediction_reference=normal_samples(0, 1, seed=3)
    )
    stable_current = {"mom": normal_samples(0, 1, seed=9), "rsi": normal_samples(50, 10, seed=10)}
    # predictions shifted far → low KS p-value → investigation (warning), not retrain
    report = await service.check_drift(
        "m1", stable_current, 1.0, 1.0, 0.60, prediction_current=normal_samples(5, 1, seed=11)
    )
    assert report.needs_investigation is True
    assert report.needs_retrain is False
    event = publisher.published[0]
    assert event.drift_type == "prediction_shift"
    assert event.severity == "warning"
    assert event.recommended_action == "alert_and_monitor"


@pytest.mark.asyncio
async def test_psi_only_computed_for_shared_features():
    service = build_service()
    service.register_baseline("m1", stable_ref(), baseline_sharpe=1.0)
    # current provides only 'mom' → 'rsi' is skipped, no crash
    report = await service.check_drift("m1", {"mom": normal_samples(0, 1, seed=9)}, 1.0, 1.0, 0.6)
    assert "mom" in report.feature_psi_scores
    assert "rsi" not in report.feature_psi_scores


def test_registry_lists_models():
    service = build_service()
    service.register_baseline("m1", stable_ref(), baseline_sharpe=1.0)
    service.register_baseline("m2", stable_ref(), baseline_sharpe=0.7)
    assert service.registry.model_ids() == ["m1", "m2"]
