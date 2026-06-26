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
    assert body["service"] == "strategy"


@pytest.mark.asyncio
async def test_ready_returns_200(client: AsyncClient):
    response = await client.get("/ready")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_metrics_endpoint_exists(client: AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_ready_503_when_nats_down(client: AsyncClient):
    from src.main import app

    async def checker():
        return False, {"nats": False}

    app.state.readiness_check = checker
    try:
        resp = await client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["checks"]["nats"] is False
    finally:
        delattr(app.state, "readiness_check")


@pytest.mark.asyncio
async def test_ready_200_when_nats_up(client: AsyncClient):
    from src.main import app

    async def checker():
        return True, {"nats": True}

    app.state.readiness_check = checker
    try:
        resp = await client.get("/ready")
        assert resp.status_code == 200
    finally:
        delattr(app.state, "readiness_check")
