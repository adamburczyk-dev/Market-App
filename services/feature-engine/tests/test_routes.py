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


@pytest.mark.asyncio
async def test_ranked_universe(wired: tuple[AsyncClient, FeatureEngineService]):
    client, _ = wired
    await client.post("/api/v1/feature-engine/compute/AAPL")
    await client.post("/api/v1/feature-engine/compute/MSFT")

    resp = await client.get("/api/v1/feature-engine/ranked")
    assert resp.status_code == 200
    vectors = resp.json()
    assert len(vectors) == 2
    assert all(v["rank_transformed"] for v in vectors)
    for v in vectors:
        assert all(0.0 <= val <= 1.0 for val in v["features"].values())


@pytest.mark.asyncio
async def test_ranked_single_symbol(wired: tuple[AsyncClient, FeatureEngineService]):
    client, _ = wired
    await client.post("/api/v1/feature-engine/compute/AAPL")
    resp = await client.get("/api/v1/feature-engine/ranked/aapl")
    assert resp.status_code == 200
    assert resp.json()["rank_transformed"] is True


@pytest.mark.asyncio
async def test_ranked_unknown_404(wired: tuple[AsyncClient, FeatureEngineService]):
    client, _ = wired
    resp = await client.get("/api/v1/feature-engine/ranked/UNKNOWN")
    assert resp.status_code == 404
