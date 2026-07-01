import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "macro-data"


@pytest.mark.asyncio
async def test_ready_returns_200(client: AsyncClient):
    resp = await client.get("/ready")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_metrics_endpoint_exists(client: AsyncClient):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
