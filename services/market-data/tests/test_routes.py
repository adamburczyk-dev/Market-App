"""Testy HTTP endpointów z w pełni podpiętym serwisem."""

import pytest
from httpx import AsyncClient

from src.core.service import MarketDataService


@pytest.mark.asyncio
async def test_fetch_then_get_ohlcv(wired: tuple[AsyncClient, MarketDataService]):
    client, _ = wired

    resp = await client.post("/api/v1/market-data/fetch/aapl")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["symbol"] == "AAPL"  # symbol jest normalizowany do uppercase
    assert body["rows"] == 3

    resp = await client.get("/api/v1/market-data/ohlcv/AAPL")
    assert resp.status_code == 200
    bars = resp.json()
    assert len(bars) == 3
    assert bars[0]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_ohlcv_empty_before_fetch(wired: tuple[AsyncClient, MarketDataService]):
    client, _ = wired
    resp = await client.get("/api/v1/market-data/ohlcv/MSFT")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_symbols_reflects_stored_data(wired: tuple[AsyncClient, MarketDataService]):
    client, _ = wired
    await client.post("/api/v1/market-data/fetch/AAPL")
    resp = await client.get("/api/v1/market-data/symbols")
    assert resp.status_code == 200
    assert resp.json()["symbols"] == ["AAPL"]


@pytest.mark.asyncio
async def test_ohlcv_limit_validation(wired: tuple[AsyncClient, MarketDataService]):
    client, _ = wired
    resp = await client.get("/api/v1/market-data/ohlcv/AAPL", params={"limit": 0})
    assert resp.status_code == 422  # limit musi być >= 1


@pytest.mark.asyncio
async def test_unexpected_failure_names_its_cause(wired: tuple[AsyncClient, MarketDataService]):
    """A storage/cache/event failure must not surface as a bare 500.

    Starlette answers unhandled exceptions with an EMPTY plain-text body, so
    the caller (e.g. the bootstrap script) printed "HTTP 500:" and nothing
    else — undiagnosable from the outside. The route names the cause instead.
    """
    client, service = wired

    async def boom(*args, **kwargs):
        raise RuntimeError('relation "ohlcv" does not exist')

    service.fetch_and_store = boom  # type: ignore[method-assign]

    resp = await client.post("/api/v1/market-data/fetch/AAPL")
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "RuntimeError" in detail
    assert 'relation "ohlcv" does not exist' in detail
