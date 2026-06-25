"""Testy HTTP endpointów feature-engine z podpiętym serwisem."""

import pytest
from httpx import AsyncClient

from src.core.service import FeatureEngineService


@pytest.mark.asyncio
async def test_compute_then_get(wired: tuple[AsyncClient, FeatureEngineService]):
    client, _ = wired

    resp = await client.post("/api/v1/feature-engine/compute/aapl")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "AAPL"  # uppercased
    assert body["tier"] == 1
    assert "rsi_14" in body["features"]

    resp = await client.get("/api/v1/feature-engine/features/AAPL")
    assert resp.status_code == 200
    assert resp.json()["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_unknown_symbol_404(wired: tuple[AsyncClient, FeatureEngineService]):
    client, _ = wired
    resp = await client.get("/api/v1/feature-engine/features/UNKNOWN")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_features_reflects_computed(wired: tuple[AsyncClient, FeatureEngineService]):
    client, _ = wired
    await client.post("/api/v1/feature-engine/compute/AAPL")
    resp = await client.get("/api/v1/feature-engine/features")
    assert resp.status_code == 200
    assert resp.json()["symbols"] == ["AAPL"]


@pytest.mark.asyncio
async def test_compute_market_data_unreachable_502(wired_failing: AsyncClient):
    resp = await wired_failing.post("/api/v1/feature-engine/compute/AAPL")
    assert resp.status_code == 502
