"""Testy HTTP endpointów execution."""

import pytest
from httpx import AsyncClient

from src.core.service import ExecutionService


@pytest.mark.asyncio
async def test_portfolio_starts_flat(wired: tuple[AsyncClient, ExecutionService]):
    client, _ = wired
    resp = await client.get("/api/v1/execution/portfolio")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cash"] == 100_000.0
    assert body["equity"] == 100_000.0
    assert body["exposure_pct"] == 0.0


@pytest.mark.asyncio
async def test_execute_then_positions(wired: tuple[AsyncClient, ExecutionService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/execution/execute",
        json={"symbol": "AAPL", "side": "BUY", "quantity": 50, "price": 100.0},
    )
    assert resp.status_code == 200
    assert resp.json()["filled_quantity"] == 50.0

    resp = await client.get("/api/v1/execution/positions")
    assert resp.json()["positions"]["AAPL"]["quantity"] == 50.0

    resp = await client.get("/api/v1/execution/portfolio")
    assert resp.json()["cash"] == 95_000.0
