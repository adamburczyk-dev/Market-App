import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_service_name(client: AsyncClient):
    response = await client.get("/health")
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "ml-pipeline"


@pytest.mark.asyncio
async def test_ready_returns_200(client: AsyncClient):
    response = await client.get("/ready")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_metrics_endpoint_exists(client: AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == 200
