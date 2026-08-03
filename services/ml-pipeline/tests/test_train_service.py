"""Service-level training path: fetch history → dataset → gate → registry."""

import numpy as np
import pytest
from trading_common.schemas import Interval

from src.core.data_contract import TrainingDataContract
from src.core.dataset import DatasetParams
from src.core.labels import LabelParams
from src.core.model_store import MlflowModelStore
from src.core.monitoring.drift_detector import DriftDetector
from src.core.registry import ModelRegistry
from src.core.service import MLPipelineService
from src.events.publisher import NullPublisher

from .test_dataset import make_bars, trending
from .test_training import SMALL

# Toy scale: 3 symbols, ~200 sessions. Production thresholds (20 names,
# 1000 sessions) exist to reject exactly this shape, so the tests state their
# assumptions explicitly instead of weakening the defaults.
# Horizon pinned EXPLICITLY: these exercise label/dataset mechanics, not the
# production target, and a 120-bar fixture cannot resolve a 63-session label.
# Pinning also makes them immune to the next D2 — what the model should aim
# at is a decision, and a decision does not belong in an algorithm's test.
TOY_PARAMS = DatasetParams(label=LabelParams(horizon=10), min_history=60, min_universe=2)
TOY_CONTRACT = TrainingDataContract(
    min_sessions=50, min_symbols_per_session=2, min_samples=50, max_missing_rate_per_feature=1.0
)


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
        data_contract=TOY_CONTRACT,
        dataset_params=TOY_PARAMS,
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
    assert result["samples"] > 0
    # The gate VERDICT is not this test's subject — a 3-name cross-section
    # cannot demonstrate rank skill and rightly fails G1/G2 (see
    # test_gate_passable_end_to_end for the verdict). What must hold here is
    # that the run is logged and reviewable whatever the verdict.
    assert isinstance(result["gate"]["passed"], bool)
    assert {c["id"] for c in result["gate"]["conditions"]} == {
        "G0",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
    }
    # drift baseline registered under the versioned id, predictions included
    baseline = service.registry.get("global_v1@v1")
    assert baseline is not None
    assert baseline.prediction_reference
    assert set(baseline.reference_features) == set(result["features"])
    # version visible in the store, not yet production
    listed = store.versions()
    assert len(listed) == 1
    assert listed[0]["version"] == "1"
    assert listed[0]["run_id"]
    assert listed[0]["production"] is False
    # when it was registered — the only field that answers "did the run that
    # just took three hours actually produce this, or is it last week's?"
    assert listed[0]["created_at"] is not None


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
    assert result["dataset"]["symbols_missing"] == ["GHOST"]  # skipped, and said so
    assert result["samples"] > 0


@pytest.mark.asyncio
async def test_train_without_market_client_raises():
    service = MLPipelineService(
        DriftDetector(),
        ModelRegistry(),
        NullPublisher(),
        data_contract=TOY_CONTRACT,
        dataset_params=TOY_PARAMS,
    )
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
        data_contract=TOY_CONTRACT,
        dataset_params=TOY_PARAMS,
    )
    result = await service.train(list(universe), Interval.D1, params=SMALL)
    assert result["version"] is None
    assert result["model_id"] == "unpersisted"
    assert result["gate"]["conditions"]  # still fully reported without a store
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


@pytest.mark.asyncio
async def test_train_refuses_a_truncated_dataset(tmp_path):
    """T0-1 in the service path: production thresholds reject a toy dataset.

    This is the cache-truncation incident in miniature — the run must fail
    loudly instead of producing a model trained on a fraction of the history.
    """
    from src.core.data_contract import TrainingDataContractError

    universe = {
        "UP": make_bars("UP", trending(220, 0.004)),
        "DOWN": make_bars("DOWN", trending(220, -0.004)),
        "FLATISH": make_bars("FLATISH", trending(220, 0.0005)),
    }
    store = MlflowModelStore(f"sqlite:///{tmp_path}/mlflow.db", model_name="global_v1")
    service = MLPipelineService(
        DriftDetector(),
        ModelRegistry(),
        NullPublisher(),
        market_client=FakeMarketDataClient(universe),
        model_store=store,
        dataset_params=TOY_PARAMS,  # default contract: 1000 sessions, 20 names
    )
    with pytest.raises(TrainingDataContractError) as exc:
        await service.train(list(universe), Interval.D1, params=SMALL)
    assert exc.value.report["passed"] is False
    assert any("sessions" in v for v in exc.value.violations)


@pytest.mark.asyncio
async def test_feature_importance_study_admits_candidates_without_adopting_them():
    """Faza 3 study: measurable, planted noise floor, and NOT a training run.

    ``include_candidates`` is the conditional half of stage E2 — the classic-TA
    block is admitted so the question "would the model USE it" can be asked at
    all. Nothing about the production contract changes: the study fits its own
    model and registers nothing.
    """
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
        data_contract=TOY_CONTRACT,
        dataset_params=TOY_PARAMS,
    )
    plain = await service.feature_importance(
        list(universe), Interval.D1, limit=1500, n_repeats=2, params=SMALL
    )
    with_candidates = await service.feature_importance(
        list(universe), Interval.D1, limit=1500, n_repeats=2, include_candidates=True, params=SMALL
    )

    assert plain["registrable"] is False
    assert plain["noise_control_planted"] is True
    assert plain["importance"]["noise_control"] is not None
    # nothing was logged to the registry by a diagnostic
    assert service.registry.model_ids() == []

    measured = {row["feature"] for row in with_candidates["importance"]["features"]}
    assert "rsi_14" in measured  # a production feature is still there
    assert measured > {row["feature"] for row in plain["importance"]["features"]}
    assert with_candidates["include_candidates"] is True
