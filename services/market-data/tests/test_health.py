"""Testy health / ready endpoints — weryfikacja startu serwisu."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_returns_correct_service(client: AsyncClient):
    response = await client.get("/health")
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "market-data"


@pytest.mark.asyncio
async def test_ready_returns_200(client: AsyncClient):
    response = await client.get("/ready")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_metrics_endpoint_exists(client: AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_list_symbols(client: AsyncClient):
    response = await client.get("/api/v1/market-data/symbols")
    assert response.status_code == 200
    body = response.json()
    assert "symbols" in body
    assert isinstance(body["symbols"], list)
    assert len(body["symbols"]) > 0


@pytest.mark.asyncio
async def test_get_ohlcv_not_implemented(client: AsyncClient):
    response = await client.get("/api/v1/market-data/ohlcv/AAPL")
    assert response.status_code == 501


@pytest.mark.asyncio
async def test_trigger_fetch_accepted(client: AsyncClient):
    response = await client.post("/api/v1/market-data/fetch/AAPL")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "accepted"
    assert body["symbol"] == "AAPL"
