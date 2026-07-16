"""Tests for the serving engine (plan ML-2): features.ready → ML vote."""

from datetime import UTC, datetime

import httpx
import pytest
from trading_common.events import EventType, FeaturesReadyEvent
from trading_common.schemas import FeatureVector, Interval

from src.core.feature_client import HttpFeatureClient
from src.core.macro_client import HttpMacroClient, NullMacroClient
from src.core.model_store import MlflowModelStore
from src.core.serving import ServingEngine
from src.events.publisher import NullPublisher

from .test_training import SMALL, synthetic_dataset


class FakeFeatureClient:
    def __init__(self, features: dict[str, float] | None) -> None:
        self.features = features

    async def get_ranked(self, symbol: str, interval: Interval) -> FeatureVector | None:
        if self.features is None:
            return None
        return FeatureVector(
            symbol=symbol,
            timestamp=datetime(2026, 7, 13, tzinfo=UTC),
            interval=interval,
            features=self.features,
            tier=1,
            rank_transformed=True,
        )

    async def aclose(self) -> None:
        return None


class FakeMacroClient:
    def __init__(self, regime: str | None = None) -> None:
        self.regime = regime

    async def get_regime(self) -> str | None:
        return self.regime

    async def aclose(self) -> None:
        return None


@pytest.fixture(scope="module")
def promoted_store(tmp_path_factory):
    """A real sqlite store with a trained, promoted model."""
    import os

    from src.core.training import run_training

    tmp = tmp_path_factory.mktemp("serving-store")
    ds = synthetic_dataset()
    model, report = run_training(ds, SMALL)
    store = MlflowModelStore(f"sqlite:///{tmp}/mlflow.db", model_name="global_v1")
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        store.promote(store.log_training(model, report))
    finally:
        os.chdir(cwd)
    return store, model, ds


def build_engine(promoted_store, features, regime=None, publisher=None, **kwargs):
    store, model, _ds = promoted_store
    engine = ServingEngine(
        publisher or NullPublisher(),
        FakeFeatureClient(features),
        FakeMacroClient(regime),
        store,
        **kwargs,
    )
    assert engine.reload() == "global_v1@v1"
    return engine


def strong_vector(model, value: float) -> dict[str, float]:
    """Every non-macro model feature set to `value` (rank-space extremes)."""
    return {n: value for n in model.feature_names if not n.startswith("macro_")}


@pytest.mark.asyncio
async def test_infer_publishes_actionable_vote(promoted_store):
    _store, model, _ds = promoted_store
    publisher = NullPublisher()
    engine = build_engine(
        promoted_store,
        strong_vector(model, 0.95),
        publisher=publisher,
        buy_threshold=0.5001,
        sell_threshold=0.4999,
    )
    event = await engine.infer_symbol("AAPL", Interval.D1)
    if event is None:  # extreme vector may land under the dead zone → try the other side
        engine2 = build_engine(
            promoted_store,
            strong_vector(model, 0.05),
            publisher=publisher,
            buy_threshold=0.5001,
            sell_threshold=0.4999,
        )
        event = await engine2.infer_symbol("AAPL", Interval.D1)
    assert event is not None
    assert event.event_type == EventType.ML_SIGNAL_GENERATED
    assert event.signal in ("BUY", "SELL")
    assert event.model_id == "global_v1@v1"
    assert event.model_stack == "global_v1"
    assert 0.0 <= event.probability_up <= 1.0
    assert 0.0 <= event.confidence <= 1.0
    assert publisher.published[-1] is event


@pytest.mark.asyncio
async def test_dead_zone_is_silent(promoted_store):
    _store, model, _ds = promoted_store
    publisher = NullPublisher()
    # impossible-to-clear thresholds → everything lands in the dead zone
    engine = build_engine(
        promoted_store,
        strong_vector(model, 0.9),
        publisher=publisher,
        buy_threshold=1.01,
        sell_threshold=-0.01,
    )
    assert await engine.infer_symbol("AAPL", Interval.D1) is None
    assert publisher.published == []


@pytest.mark.asyncio
async def test_inactive_engine_is_silent(promoted_store):
    _store, model, _ds = promoted_store
    engine = ServingEngine(
        NullPublisher(), FakeFeatureClient(strong_vector(model, 0.9)), FakeMacroClient(), None
    )
    assert engine.reload() is None
    assert not engine.active
    assert await engine.infer_symbol("AAPL", Interval.D1) is None


@pytest.mark.asyncio
async def test_missing_symbol_vector_is_noop(promoted_store):
    engine = build_engine(promoted_store, None)  # feature client returns None
    assert await engine.infer_symbol("GHOST", Interval.D1) is None


@pytest.mark.asyncio
async def test_majority_feature_mismatch_refuses(promoted_store):
    publisher = NullPublisher()
    engine = build_engine(
        promoted_store,
        {"totally_unrelated": 0.5},
        publisher=publisher,
        buy_threshold=0.5001,
        sell_threshold=0.4999,
    )
    assert await engine.infer_symbol("AAPL", Interval.D1) is None  # refusal, not 0.5-fill
    assert publisher.published == []


@pytest.mark.asyncio
async def test_sparse_tier2_attributes_fill_neutral(promoted_store):
    _store, model, _ds = promoted_store
    features = strong_vector(model, 0.9)
    dropped = sorted(features)[0]
    features.pop(dropped)  # one missing attr ≠ schema drift → neutral fill
    engine = build_engine(promoted_store, features, buy_threshold=0.5001, sell_threshold=0.4999)
    row = engine._assemble(features, None)
    assert row is not None
    names = model.feature_names
    assert row[names.index(dropped)] == 0.5


def test_macro_one_hot_in_assembly(promoted_store):
    _store, model, _ds = promoted_store
    engine = build_engine(promoted_store, strong_vector(model, 0.9), regime="crisis")
    row = engine._assemble(strong_vector(model, 0.9), "crisis")
    names = model.feature_names
    assert row[names.index("macro_crisis")] == 1.0
    assert row[names.index("macro_expansion")] == 0.0


@pytest.mark.asyncio
async def test_handler_filters_other_intervals(promoted_store):
    _store, model, _ds = promoted_store
    publisher = NullPublisher()
    engine = build_engine(
        promoted_store,
        strong_vector(model, 0.95),
        publisher=publisher,
        buy_threshold=0.5001,
        sell_threshold=0.4999,
    )
    hourly = FeaturesReadyEvent(symbol="AAPL", interval="1h", features_count=10)
    await engine.handle_features_ready(hourly.model_dump_json().encode())
    assert publisher.published == []  # only the configured serve interval infers


# --- HTTP clients ---


@pytest.mark.asyncio
async def test_feature_client_parses_and_404s():
    def handler(request: httpx.Request) -> httpx.Response:
        if "GHOST" in request.url.path:
            return httpx.Response(404)
        fv = FeatureVector(
            symbol="AAPL",
            timestamp=datetime(2026, 7, 13, tzinfo=UTC),
            interval=Interval.D1,
            features={"momentum_20": 0.9},
            tier=1,
            rank_transformed=True,
        )
        return httpx.Response(200, json=fv.model_dump(mode="json"))

    client = HttpFeatureClient("http://feature-engine:8000")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fv = await client.get_ranked("AAPL", Interval.D1)
    assert fv is not None and fv.features["momentum_20"] == 0.9
    assert await client.get_ranked("GHOST", Interval.D1) is None
    await client.aclose()


@pytest.mark.asyncio
async def test_macro_client_caches_within_ttl():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"regime": "expansion"})

    now = [0.0]
    client = HttpMacroClient("http://macro-data:8000", cache_ttl_s=600.0, clock=lambda: now[0])
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await client.get_regime() == "expansion"
    assert await client.get_regime() == "expansion"
    assert len(calls) == 1  # served from cache
    now[0] = 601.0
    assert await client.get_regime() == "expansion"
    assert len(calls) == 2  # TTL elapsed → re-fetched
    await client.aclose()


@pytest.mark.asyncio
async def test_macro_client_degrades_to_none():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("macro-data down")

    client = HttpMacroClient("http://macro-data:8000")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    assert await client.get_regime() is None
    await client.aclose()


@pytest.mark.asyncio
async def test_null_macro_client():
    assert await NullMacroClient().get_regime() is None


@pytest.mark.asyncio
async def test_promote_hot_reloads_serving(promoted_store, tmp_path, monkeypatch):
    """service.promote() must swap the serving model without a restart."""

    from src.core.monitoring.drift_detector import DriftDetector
    from src.core.registry import ModelRegistry
    from src.core.service import MLPipelineService
    from src.core.training import run_training

    store, model, ds = promoted_store
    engine = ServingEngine(NullPublisher(), FakeFeatureClient({}), FakeMacroClient(), store)
    assert engine.reload() == "global_v1@v1"

    monkeypatch.chdir(tmp_path)
    _, report = run_training(ds, SMALL)
    v2 = store.log_training(model, report)
    service = MLPipelineService(
        DriftDetector(), ModelRegistry(), NullPublisher(), model_store=store, serving=engine
    )
    result = service.promote(v2)
    assert result["serving_model_id"] == f"global_v1@v{v2}"
    assert engine.model_id == f"global_v1@v{v2}"
