"""Tests for macro-data HTTP routes."""

import pytest
from httpx import AsyncClient

from src.core.service import MacroDataService


@pytest.mark.asyncio
async def test_status_ok(client: AsyncClient):
    resp = await client.get("/api/v1/macro-data/status")
    assert resp.status_code == 200
    assert resp.json()["service"] == "macro-data"


@pytest.mark.asyncio
async def test_snapshot_404_before_refresh(wired: tuple[AsyncClient, MacroDataService]):
    client, _ = wired
    resp = await client.get("/api/v1/macro-data/snapshot")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_refresh_then_snapshot_and_regime(wired: tuple[AsyncClient, MacroDataService]):
    client, _ = wired
    r = await client.post(
        "/api/v1/macro-data/refresh",
        json={"yield_curve_10y_2y": 1.5, "credit_spread_baa_10y": 1.2, "pmi": 57},
    )
    assert r.status_code == 200
    assert r.json()["regime"] == "expansion"

    snap = await client.get("/api/v1/macro-data/snapshot")
    assert snap.status_code == 200
    assert snap.json()["regime"] == "expansion"

    reg = await client.get("/api/v1/macro-data/regime")
    assert reg.json()["regime"] == "expansion"


@pytest.mark.asyncio
async def test_regime_none_before_refresh(wired: tuple[AsyncClient, MacroDataService]):
    client, _ = wired
    resp = await client.get("/api/v1/macro-data/regime")
    assert resp.status_code == 200
    assert resp.json()["regime"] is None


@pytest.mark.asyncio
async def test_refresh_crisis_regime(wired: tuple[AsyncClient, MacroDataService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/macro-data/refresh",
        json={"yield_curve_10y_2y": -0.8, "credit_spread_baa_10y": 3.5, "pmi": 44},
    )
    assert resp.json()["regime"] == "crisis"


@pytest.mark.asyncio
async def test_snapshot_503_when_unwired(client: AsyncClient):
    resp = await client.get("/api/v1/macro-data/snapshot")
    assert resp.status_code == 503
