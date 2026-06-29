"""Tests for ml-pipeline HTTP routes (service wired via dependency override)."""

import pytest
from httpx import AsyncClient

from src.core.service import MLPipelineService

from .conftest import normal_samples


def baseline_body() -> dict:
    return {
        "reference_features": {
            "mom": normal_samples(0, 1, seed=1),
            "rsi": normal_samples(50, 10, seed=2),
        },
        "baseline_sharpe": 1.0,
    }


@pytest.mark.asyncio
async def test_status_ok(client: AsyncClient):
    resp = await client.get("/api/v1/ml-pipeline/status")
    assert resp.status_code == 200
    assert resp.json()["service"] == "ml-pipeline"


@pytest.mark.asyncio
async def test_register_then_list(wired: tuple[AsyncClient, MLPipelineService]):
    client, _ = wired
    resp = await client.post("/api/v1/ml-pipeline/models/m1/baseline", json=baseline_body())
    assert resp.status_code == 200
    assert resp.json()["registered"] is True

    listing = await client.get("/api/v1/ml-pipeline/models")
    assert listing.json()["models"] == ["m1"]


@pytest.mark.asyncio
async def test_drift_check_reports_feature_drift(wired: tuple[AsyncClient, MLPipelineService]):
    client, _ = wired
    await client.post("/api/v1/ml-pipeline/models/m1/baseline", json=baseline_body())
    resp = await client.post(
        "/api/v1/ml-pipeline/models/m1/drift",
        json={
            "current_features": {"mom": normal_samples(3, 1, seed=9)},
            "rolling_sharpe_30d": 1.0,
            "rolling_sharpe_90d": 1.0,
            "rolling_accuracy_30d": 0.6,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["needs_retrain"] is True
    assert "mom" in body["features_drifted"]
    assert body["recommended_action"] == "auto_retrain"


@pytest.mark.asyncio
async def test_drift_check_unknown_model_404(wired: tuple[AsyncClient, MLPipelineService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/ml-pipeline/models/ghost/drift",
        json={
            "current_features": {"mom": [1.0, 2.0]},
            "rolling_sharpe_30d": 1.0,
            "rolling_sharpe_90d": 1.0,
            "rolling_accuracy_30d": 0.6,
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_drift_check_503_when_unwired(client: AsyncClient):
    resp = await client.post(
        "/api/v1/ml-pipeline/models/m1/drift",
        json={
            "current_features": {"mom": [1.0]},
            "rolling_sharpe_30d": 1.0,
            "rolling_sharpe_90d": 1.0,
            "rolling_accuracy_30d": 0.6,
        },
    )
    assert resp.status_code == 503
