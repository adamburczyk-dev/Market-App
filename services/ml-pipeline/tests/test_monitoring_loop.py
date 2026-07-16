"""Tests for the ML-3 monitoring loop: inference log → outcomes → daily drift."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.core.inference_log import InferenceLog, InferenceRecord
from src.core.labels import LabelParams
from src.core.monitoring.drift_detector import DriftDetector
from src.core.outcomes import OutcomeResolver
from src.core.registry import ModelRegistry
from src.core.service import MLPipelineService
from src.events.publisher import NullPublisher

from .test_dataset import make_bars

MODEL = "global_v1@v1"
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def record(symbol="AAPL", signal="BUY", p=0.7, at=NOW, features=None):
    return InferenceRecord(
        symbol=symbol,
        at=at,
        features=features or {"momentum_20": 0.9, "rsi_14": 0.6},
        probability_up=p,
        signal=signal,
    )


# --- InferenceLog ---


def test_log_windows_and_counts():
    log = InferenceLog(maxlen=5)
    for i in range(8):  # bounded: only the last 5 survive
        log.append(MODEL, record(p=0.1 * i, signal="HOLD"))
    assert log.counts(MODEL)["total"] == 5
    assert log.prediction_window(MODEL) == pytest.approx([0.3, 0.4, 0.5, 0.6, 0.7])
    assert set(log.feature_window(MODEL)) == {"momentum_20", "rsi_14"}
    assert log.pending(MODEL) == []  # HOLDs never await outcomes


def test_pending_and_resolution():
    log = InferenceLog()
    r1, r2 = record(signal="BUY"), record(signal="SELL")
    log.append(MODEL, r1)
    log.append(MODEL, r2)
    log.append(MODEL, record(signal="HOLD"))
    assert len(log.pending(MODEL)) == 2
    log.resolve(r1, 1, 0.03, True, NOW)
    assert len(log.pending(MODEL)) == 1
    assert log.counts(MODEL)["resolved"] == 1


def test_rolling_metrics_need_min_outcomes():
    log = InferenceLog()
    for _ in range(5):
        r = record()
        log.append(MODEL, r)
        log.resolve(r, 1, 0.02, True, NOW)
    assert log.rolling_metrics(MODEL, 30, 10, min_outcomes=10, now=NOW) is None
    metrics = log.rolling_metrics(MODEL, 30, 10, min_outcomes=5, now=NOW)
    assert metrics is not None
    _sharpe, accuracy = metrics
    assert accuracy == 1.0


def test_rolling_metrics_window_and_sign():
    log = InferenceLog()
    old = NOW - timedelta(days=90)
    for _ in range(10):  # old wins outside the 30d window
        r = record()
        log.append(MODEL, r)
        log.resolve(r, 1, 0.05, True, old)
    for k in range(10):  # recent: half losses
        r = record()
        log.append(MODEL, r)
        log.resolve(r, 0, -0.02 if k % 2 else 0.02, k % 2 == 0, NOW)
    sharpe, accuracy = log.rolling_metrics(MODEL, 30, 10, min_outcomes=10, now=NOW)
    assert accuracy == 0.5
    assert abs(sharpe) < 5  # mixed returns → modest sharpe, correct windowing


# --- OutcomeResolver ---


class HistoryMarket:
    """Serves a deterministic OHLCV path ending at `NOW`'s date."""

    def __init__(self, closes: list[float]) -> None:
        start = NOW - timedelta(days=len(closes) - 1)
        self.bars = make_bars("AAPL", closes)
        for i, bar in enumerate(self.bars):  # re-stamp so dates align with NOW
            object.__setattr__(bar, "timestamp", start + timedelta(days=i))

    async def get_ohlcv(self, symbol, interval, limit=500):  # type: ignore[no-untyped-def]
        return self.bars[-limit:]

    async def aclose(self) -> None:
        return None


def up_path(n: int = 60) -> list[float]:
    # noisy base then a hard rally at the end — the up barrier gets touched
    base = [100 + (0.5 if i % 2 else -0.5) for i in range(n - 12)]
    return base + [base[-1] * (1 + 0.03 * k) for k in range(1, 13)]


@pytest.mark.asyncio
async def test_buy_on_up_path_resolves_correct_and_positive():
    log = InferenceLog()
    vote_at = NOW - timedelta(days=11)  # mature: 11 days of path after the vote
    r = record(signal="BUY", at=vote_at)
    log.append(MODEL, r)
    resolver = OutcomeResolver(HistoryMarket(up_path()), log, LabelParams())
    resolved = await resolver.resolve_pending(MODEL, now=NOW)
    assert len(resolved) == 1
    assert resolved[0] > 0
    assert r.resolved and r.correct is True and r.label == 1


@pytest.mark.asyncio
async def test_sell_on_up_path_resolves_incorrect_and_negative():
    log = InferenceLog()
    r = record(signal="SELL", at=NOW - timedelta(days=11))
    log.append(MODEL, r)
    resolver = OutcomeResolver(HistoryMarket(up_path()), log, LabelParams())
    resolved = await resolver.resolve_pending(MODEL, now=NOW)
    assert resolved[0] < 0
    assert r.correct is False


@pytest.mark.asyncio
async def test_immature_vote_stays_pending():
    log = InferenceLog()
    r = record(signal="BUY", at=NOW)  # voted today — no future path yet
    log.append(MODEL, r)
    # calm path: barriers never touched → vertical needs the full horizon
    calm = [100 + (0.4 if i % 2 else -0.4) for i in range(60)]
    resolver = OutcomeResolver(HistoryMarket(calm), log, LabelParams())
    assert await resolver.resolve_pending(MODEL, now=NOW) == []
    assert not r.resolved
    assert len(log.pending(MODEL)) == 1


@pytest.mark.asyncio
async def test_stale_unresolvable_vote_is_dropped():
    log = InferenceLog()
    r = record(signal="BUY", at=NOW - timedelta(days=60))  # older than drop_after
    log.append(MODEL, r)
    calm = [100 + (0.4 if i % 2 else -0.4) for i in range(30)]  # history misses the entry
    resolver = OutcomeResolver(HistoryMarket(calm), log, LabelParams(), drop_after_days=42)
    assert await resolver.resolve_pending(MODEL, now=NOW) == []
    assert r.resolved and r.label is None  # dropped, no fabricated outcome
    assert log.pending(MODEL) == []


# --- run_daily_monitor (service) ---


class FakeServing:
    def __init__(self, model_id=MODEL, active=True):
        self._model_id = model_id
        self._active = active
        self.paused = False

    @property
    def active(self):
        return self._active

    @property
    def model_id(self):
        return self._model_id


class RecordingAggregator:
    def __init__(self):
        self.outcomes = []

    async def record_outcome(self, source, daily_return):
        self.outcomes.append((source, daily_return))
        return True

    async def aclose(self):
        return None


def build_monitor_service(log, publisher=None, aggregator=None, resolver=None):
    registry = ModelRegistry()
    service = MLPipelineService(
        DriftDetector(),
        registry,
        publisher or NullPublisher(),
        serving=FakeServing(),  # type: ignore[arg-type]
        inference_log=log,
        resolver=resolver,
        aggregator_client=aggregator,
    )
    return service, registry


def fill_log(log, values, n=60):
    for value in np.linspace(*values, n):
        log.append(MODEL, record(signal="HOLD", p=0.5, features={"momentum_20": float(value)}))


@pytest.mark.asyncio
async def test_monitor_publishes_drift_on_shifted_features():
    log = InferenceLog()
    fill_log(log, (0.8, 1.0))  # live window far from the baseline distribution
    publisher = NullPublisher()
    service, _ = build_monitor_service(log, publisher=publisher)
    service.register_baseline(
        MODEL, {"momentum_20": list(np.linspace(0.0, 0.3, 200))}, baseline_sharpe=1.0
    )
    result = await service.run_daily_monitor()
    assert result["recommended_action"] == "auto_retrain"
    assert any(e.event_type.value == "ml.drift_detected" for e in publisher.published)


@pytest.mark.asyncio
async def test_monitor_quiet_on_healthy_window():
    log = InferenceLog()
    fill_log(log, (0.0, 0.3))  # same distribution as the baseline
    publisher = NullPublisher()
    service, _ = build_monitor_service(log, publisher=publisher)
    service.register_baseline(
        MODEL, {"momentum_20": list(np.linspace(0.0, 0.3, 200))}, baseline_sharpe=1.0
    )
    result = await service.run_daily_monitor()
    assert result["recommended_action"] == "no_action"
    assert result["performance_measured"] is False  # neutral inputs, honestly flagged
    assert publisher.published == []


@pytest.mark.asyncio
async def test_monitor_pushes_resolved_outcomes_to_aggregator():
    log = InferenceLog()
    fill_log(log, (0.0, 0.3))
    vote = record(signal="BUY", at=NOW - timedelta(days=11))
    log.append(MODEL, vote)
    aggregator = RecordingAggregator()
    resolver = OutcomeResolver(HistoryMarket(up_path()), log, LabelParams())
    service, _ = build_monitor_service(log, aggregator=aggregator, resolver=resolver)
    service.register_baseline(
        MODEL, {"momentum_20": list(np.linspace(0.0, 0.3, 200))}, baseline_sharpe=1.0
    )
    result = await service.run_daily_monitor()
    assert result["outcomes_resolved"] == 1
    assert aggregator.outcomes and aggregator.outcomes[0][0] == "ml"
    assert aggregator.outcomes[0][1] > 0  # BUY on an up path → positive realized return


@pytest.mark.asyncio
async def test_monitor_skips_when_serving_inactive():
    service = MLPipelineService(
        DriftDetector(),
        ModelRegistry(),
        NullPublisher(),
        serving=FakeServing(active=False),  # type: ignore[arg-type]
        inference_log=InferenceLog(),
    )
    assert (await service.run_daily_monitor())["skipped"] == "serving_inactive"


@pytest.mark.asyncio
async def test_monitor_skips_without_baseline():
    log = InferenceLog()
    fill_log(log, (0.0, 0.3))
    service, _ = build_monitor_service(log)
    assert (await service.run_daily_monitor())["skipped"] == "no_baseline"
