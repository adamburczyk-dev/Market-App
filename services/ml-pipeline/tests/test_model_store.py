"""MLflow model-store round-trip on a local sqlite backend."""

import numpy as np
import pytest

from src.core.model import TrainConfig, train_classifier
from src.core.model_store import MlflowModelStore
from src.core.training import run_training

from .test_training import SMALL, synthetic_dataset


@pytest.fixture(scope="module")
def trained():
    ds = synthetic_dataset()
    model, report = run_training(ds, SMALL)
    return ds, model, report


@pytest.fixture()
def store(tmp_path):
    return MlflowModelStore(f"sqlite:///{tmp_path}/mlflow.db", model_name="global_v1")


def test_log_promote_load_round_trip(store, trained, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # mlflow writes ./mlruns artifacts relative to cwd
    ds, model, report = trained
    version = store.log_training(model, report)
    assert version == "1"
    assert store.production_version() is None  # promotion is manual

    store.promote(version)
    assert store.production_version() == "1"

    loaded, metadata = store.load_production()
    assert loaded.feature_names == model.feature_names
    assert loaded.temperature == pytest.approx(model.temperature)
    probe = ds.x[:16]
    assert np.allclose(loaded.predict_proba(probe), model.predict_proba(probe), atol=1e-6)
    assert metadata["gate"]["holdout"]["sharpe"] == report.as_dict()["holdout"]["sharpe"]


def test_versions_listing(store, trained, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, model, report = trained
    v1 = store.log_training(model, report)
    v2 = store.log_training(model, report)
    store.promote(v2)
    versions = store.versions()
    assert [v["version"] for v in versions] == [v1, v2]
    assert [v["production"] for v in versions] == [False, True]


def test_load_production_none_when_nothing_promoted(store):
    assert store.load_production() is None


def test_serving_refuses_on_feature_mismatch_is_callers_contract(
    store, trained, tmp_path, monkeypatch
):
    """The metadata is load-bearing: the feature list survives the round trip
    exactly, so the (ML-2) serving path can compare vectors key-for-key."""
    monkeypatch.chdir(tmp_path)
    _, model, report = trained
    store.promote(store.log_training(model, report))
    loaded, metadata = store.load_production()
    assert metadata["feature_names"] == model.feature_names
    assert loaded.feature_names == metadata["feature_names"]


def test_state_dict_round_trip_without_registry(tmp_path):
    """Weights+temperature reconstruction is exact for a fresh tiny model."""
    rng = np.random.default_rng(5)
    x, y = rng.uniform(size=(120, 3)), (rng.uniform(size=120) > 0.5).astype(float)
    model = train_classifier(
        x[:80],
        y[:80],
        x[80:],
        y[80:],
        ["a", "b", "c"],
        TrainConfig(hidden=(8, 4), max_epochs=5, patience=2),
    )
    store = MlflowModelStore(f"sqlite:///{tmp_path}/m.db", model_name="tiny")
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        from src.core.evaluation import PortfolioResult
        from src.core.training import FoldReport, GateReport

        empty = FoldReport("holdout", 80, 40, 0.5, 0.25, PortfolioResult(0, 0, 0, 0, 0))
        version = store.log_training(model, GateReport([], empty, False, ["synthetic"]))
        loaded, _ = store.load(version)
        assert np.allclose(loaded.predict_proba(x[:10]), model.predict_proba(x[:10]), atol=1e-6)
    finally:
        os.chdir(cwd)
