"""Tests for fundamental-data HTTP routes."""

import pytest
from httpx import AsyncClient

from src.core.service import FundamentalDataService

from .conftest import improving_pair


def ingest_body() -> dict:
    current, prior = improving_pair()
    return {
        "current": current.model_dump(mode="json"),
        "prior": prior.model_dump(mode="json"),
    }


@pytest.mark.asyncio
async def test_status_ok(client: AsyncClient):
    resp = await client.get("/api/v1/fundamental-data/status")
    assert resp.status_code == 200
    assert resp.json()["service"] == "fundamental-data"


@pytest.mark.asyncio
async def test_ingest_then_get(wired: tuple[AsyncClient, FundamentalDataService]):
    client, _ = wired
    resp = await client.post("/api/v1/fundamental-data/statements", json=ingest_body())
    assert resp.status_code == 200
    body = resp.json()
    assert body["f_score"] == 9
    assert body["f_score_breakdown"]["max_score"] == 9
    assert body["statement"]["piotroski_f_score"] == 9

    got = await client.get("/api/v1/fundamental-data/fundamentals/AAPL")
    assert got.status_code == 200
    assert got.json()["f_score"] == 9

    listing = await client.get("/api/v1/fundamental-data/fundamentals")
    assert listing.json()["symbols"] == ["AAPL"]


@pytest.mark.asyncio
async def test_get_unknown_symbol_404(wired: tuple[AsyncClient, FundamentalDataService]):
    client, _ = wired
    resp = await client.get("/api/v1/fundamental-data/fundamentals/TSLA")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_refresh_404_without_edgar_data(wired: tuple[AsyncClient, FundamentalDataService]):
    client, _ = wired
    # default FakeFetcher returns [] → 404
    resp = await client.post("/api/v1/fundamental-data/refresh/AAPL")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_statements_503_when_unwired(client: AsyncClient):
    resp = await client.post("/api/v1/fundamental-data/statements", json=ingest_body())
    assert resp.status_code == 503
