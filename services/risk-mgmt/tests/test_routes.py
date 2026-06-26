"""Testy HTTP endpointów risk-mgmt."""

import pytest
from httpx import AsyncClient

from src.core.service import RiskMgmtService


@pytest.mark.asyncio
async def test_get_portfolio(wired: tuple[AsyncClient, RiskMgmtService]):
    client, _ = wired
    resp = await client.get("/api/v1/risk-mgmt/portfolio")
    assert resp.status_code == 200
    assert resp.json()["value"] == 100000.0


@pytest.mark.asyncio
async def test_post_portfolio_trips_breaker(wired: tuple[AsyncClient, RiskMgmtService]):
    client, _ = wired
    resp = await client.post("/api/v1/risk-mgmt/portfolio", json={"daily_loss_pct": 0.06})
    assert resp.status_code == 200
    body = resp.json()
    assert body["circuit_breaker"]["tripped"] is True
    assert body["breaker_changed"] is True


@pytest.mark.asyncio
async def test_circuit_breaker_status(wired: tuple[AsyncClient, RiskMgmtService]):
    client, _ = wired
    resp = await client.get("/api/v1/risk-mgmt/circuit-breaker")
    assert resp.status_code == 200
    assert resp.json()["tripped"] is False


@pytest.mark.asyncio
async def test_process_signal_returns_order(wired: tuple[AsyncClient, RiskMgmtService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/risk-mgmt/signal",
        json={
            "symbol": "AAPL",
            "signal": "BUY",
            "price": 100.0,
            "stop_loss": 95.0,
            "strategy_name": "momentum_rank",
        },
    )
    assert resp.status_code == 200
    order = resp.json()["order"]
    assert order is not None
    assert order["side"] == "BUY"
    assert order["quantity"] == 50.0


@pytest.mark.asyncio
async def test_process_signal_hold_returns_no_order(wired: tuple[AsyncClient, RiskMgmtService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/risk-mgmt/signal",
        json={"symbol": "AAPL", "signal": "HOLD", "price": 100.0},
    )
    assert resp.status_code == 200
    assert resp.json()["order"] is None
