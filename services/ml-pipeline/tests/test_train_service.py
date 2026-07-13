"""Service-level training path: fetch history → dataset → gate → registry."""

import numpy as np
import pytest
from trading_common.schemas import Interval

from src.core.model_store import MlflowModelStore
from src.core.monitoring.drift_detector import DriftDetector
from src.core.registry import ModelRegistry
from src.core.service import MLPipelineService
from src.events.publisher import NullPublisher

from .test_dataset import make_bars, trending
from .test_training import SMALL


class FakeMarketDataClient:
    def __init__(self, universe: dict[str, list]) -> None:
        self.universe = universe
        self.calls: list[str] = []

    async def get_ohlcv(self, symbol, interval, limit=500):  # type: ignore[no-untyped-def]
        self.calls.append(symbol)
        return self.universe.get(symbol, [])[-limit:]

    async def aclose(self) -> None:
        return None


def build_service(tmp_path, universe):
    store = MlflowModelStore(f"sqlite:///{tmp_path}/mlflow.db", model_name="global_v1")
    service = MLPipelineService(
        DriftDetector(),
        ModelRegistry(),
        NullPublisher(),
        market_client=FakeMarketDataClient(universe),
        model_store=store,
    )
    return service, store


@pytest.mark.asyncio
async def test_train_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    universe = {
        "UP": make_bars("UP", trending(220, 0.004)),
        "DOWN": make_bars("DOWN", trending(220, -0.004)),
        "FLATISH": make_bars("FLATISH", trending(220, 0.0005)),
    }
    service, store = build_service(tmp_path, universe)
    result = await service.train(list(universe), Interval.D1, limit=1500, params=SMALL)

    assert result["version"] == "1"
    assert result["model_id"] == "global_v1@v1"
    assert result["gate"]["passed"] is True
    assert result["samples"] > 0
    # drift baseline registered under the versioned id, predictions included
    baseline = service.registry.get("global_v1@v1")
    assert baseline is not None
    assert baseline.prediction_reference
    assert set(baseline.reference_features) == set(result["features"])
    # version visible in the store, not yet production
    assert store.versions()[0] == {
        "version": "1",
        "run_id": store.versions()[0]["run_id"],
        "production": False,
    }


@pytest.mark.asyncio
async def test_train_skips_symbols_without_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    universe = {
        "UP": make_bars("UP", trending(220, 0.004)),
        "DOWN": make_bars("DOWN", trending(220, -0.004)),
        "FLATISH": make_bars("FLATISH", trending(220, 0.0005)),
    }
    service, _ = build_service(tmp_path, universe)
    result = await service.train([*universe, "GHOST"], Interval.D1, limit=1500, params=SMALL)
    assert result["gate"]["passed"] is True  # GHOST silently skipped


@pytest.mark.asyncio
async def test_train_without_market_client_raises():
    service = MLPipelineService(DriftDetector(), ModelRegistry(), NullPublisher())
    with pytest.raises(RuntimeError, match="market-data client"):
        await service.train(["A", "B"], Interval.D1)


@pytest.mark.asyncio
async def test_train_without_store_still_reports(tmp_path):
    universe = {
        "UP": make_bars("UP", trending(220, 0.004)),
        "DOWN": make_bars("DOWN", trending(220, -0.004)),
        "FLATISH": make_bars("FLATISH", trending(220, 0.0005)),
    }
    service = MLPipelineService(
        DriftDetector(),
        ModelRegistry(),
        NullPublisher(),
        market_client=FakeMarketDataClient(universe),
        model_store=None,
    )
    result = await service.train(list(universe), Interval.D1, params=SMALL)
    assert result["version"] is None
    assert result["model_id"] == "unpersisted"
    assert result["gate"]["passed"] is True
    assert service.registry.get("unpersisted") is not None


@pytest.mark.asyncio
async def test_promote_route_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    universe = {
        "UP": make_bars("UP", trending(220, 0.004)),
        "DOWN": make_bars("DOWN", trending(220, -0.004)),
    }
    service, store = build_service(tmp_path, universe)
    await service.train(list(universe), Interval.D1, params=SMALL)
    store.promote("1")
    loaded = store.load_production()
    assert loaded is not None
    model, metadata = loaded
    probe = np.full((2, len(model.feature_names)), 0.5)
    probs = model.predict_proba(probe)
    assert probs.shape == (2,)
    assert metadata["feature_names"] == model.feature_names
