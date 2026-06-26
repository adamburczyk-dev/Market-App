"""Testy health / ready / metrics / symbols — weryfikacja startu serwisu."""

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
async def test_list_symbols_falls_back_to_defaults(client: AsyncClient):
    """Bez podpiętego serwisu /symbols zwraca domyślne symbole."""
    response = await client.get("/api/v1/market-data/symbols")
    assert response.status_code == 200
    body = response.json()
    assert "symbols" in body
    assert isinstance(body["symbols"], list)
    assert len(body["symbols"]) > 0


@pytest.mark.asyncio
async def test_ohlcv_without_service_returns_503(client: AsyncClient):
    """Endpoint wymagający serwisu zwraca 503, gdy serwis nie jest gotowy."""
    response = await client.get("/api/v1/market-data/ohlcv/AAPL")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_ready_503_when_database_down(client: AsyncClient):
    from src.main import app

    async def checker():
        return False, {"database": False, "redis": True, "nats": True}

    app.state.readiness_check = checker
    try:
        resp = await client.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["database"] is False
    finally:
        delattr(app.state, "readiness_check")


@pytest.mark.asyncio
async def test_ready_200_with_checks_when_healthy(client: AsyncClient):
    from src.main import app

    async def checker():
        return True, {"database": True, "redis": True, "nats": True}

    app.state.readiness_check = checker
    try:
        resp = await client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["checks"]["database"] is True
    finally:
        delattr(app.state, "readiness_check")
