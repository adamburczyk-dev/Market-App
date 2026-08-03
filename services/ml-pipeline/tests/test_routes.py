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


@pytest.mark.asyncio
async def test_capacity_probe_needs_a_market_client(wired: tuple[AsyncClient, MLPipelineService]):
    # No market-data client wired → 503, not a stack trace. The probe is an ops
    # call like training and fails the same way.
    client, _ = wired
    resp = await client.post(
        "/api/v1/ml-pipeline/models/capacity-probe",
        json={"symbols": ["AAPL", "MSFT"], "interval": "1d", "limit": 500},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_sector_study_needs_a_market_client(wired: tuple[AsyncClient, MLPipelineService]):
    # Same ops-call contract as the probe: 503, not a stack trace.
    client, _ = wired
    resp = await client.post(
        "/api/v1/ml-pipeline/models/sector-study",
        json={
            "symbols": ["AAPL", "MSFT"],
            "interval": "1d",
            "limit": 500,
            "sectors": {"AAPL": "Technology", "MSFT": "Information Technology"},
        },
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_sector_study_accepts_an_absent_sector_map(
    wired: tuple[AsyncClient, MLPipelineService],
):
    # "I have no sectors" is a real request — every name lands in the residual
    # group and the study says so — so the field must not be required.
    client, _ = wired
    resp = await client.post(
        "/api/v1/ml-pipeline/models/sector-study",
        json={"symbols": ["AAPL", "MSFT"], "interval": "1d", "limit": 500},
    )
    assert resp.status_code == 503  # still unwired, but the body validated


@pytest.mark.asyncio
async def test_cost_study_needs_a_market_client(wired: tuple[AsyncClient, MLPipelineService]):
    # Same ops-call contract as the other studies: 503, not a stack trace.
    client, _ = wired
    resp = await client.post(
        "/api/v1/ml-pipeline/models/cost-study",
        json={"symbols": ["AAPL", "MSFT"], "interval": "1d", "limit": 500, "aum_usd": 5_000_000},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_cost_study_defaults_the_book_size(wired: tuple[AsyncClient, MLPipelineService]):
    """A cost number is meaningless without a size, so the size must always be
    present — defaulted rather than optional-and-absent."""
    client, _ = wired
    resp = await client.post(
        "/api/v1/ml-pipeline/models/cost-study",
        json={"symbols": ["AAPL", "MSFT"], "interval": "1d", "limit": 500},
    )
    assert resp.status_code == 503  # unwired, but the body validated
    bad = await client.post(
        "/api/v1/ml-pipeline/models/cost-study",
        json={"symbols": ["AAPL", "MSFT"], "interval": "1d", "limit": 500, "aum_usd": 0},
    )
    assert bad.status_code == 422


@pytest.mark.asyncio
async def test_feature_importance_needs_a_market_client(
    wired: tuple[AsyncClient, MLPipelineService],
):
    client, _ = wired
    resp = await client.post(
        "/api/v1/ml-pipeline/models/feature-importance",
        json={"symbols": ["AAPL", "MSFT"], "interval": "1d", "limit": 500},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_feature_importance_refuses_zero_permutations(
    wired: tuple[AsyncClient, MLPipelineService],
):
    """Zero repeats is not a cheaper study, it is no measurement at all — and a
    report shaped like the real one with nothing behind it is worse than a 422."""
    client, _ = wired
    ok = await client.post(
        "/api/v1/ml-pipeline/models/feature-importance",
        json={"symbols": ["AAPL", "MSFT"], "interval": "1d", "limit": 500},
    )
    assert ok.status_code == 503  # unwired, but the body validated
    bad = await client.post(
        "/api/v1/ml-pipeline/models/feature-importance",
        json={"symbols": ["AAPL", "MSFT"], "interval": "1d", "limit": 500, "n_repeats": 0},
    )
    assert bad.status_code == 422


@pytest.mark.asyncio
async def test_alpha_decay_needs_a_market_client(wired: tuple[AsyncClient, MLPipelineService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/ml-pipeline/models/alpha-decay",
        json={"symbols": ["AAPL", "MSFT"], "interval": "1d", "limit": 500},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_alpha_decay_refuses_an_empty_horizon_list(
    wired: tuple[AsyncClient, MLPipelineService],
):
    """A decay profile with no horizons is not a smaller study, it is no study
    — better a 422 than a report with an empty table."""
    client, _ = wired
    resp = await client.post(
        "/api/v1/ml-pipeline/models/alpha-decay",
        json={"symbols": ["AAPL", "MSFT"], "interval": "1d", "limit": 500, "horizons": []},
    )
    assert resp.status_code == 422
