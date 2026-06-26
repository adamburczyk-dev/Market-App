"""Testy HTTP endpointów strategy."""

import pytest
from httpx import AsyncClient

from src.core.service import StrategyService


@pytest.mark.asyncio
async def test_status(wired: tuple[AsyncClient, StrategyService]):
    client, _ = wired
    resp = await client.get("/api/v1/strategy/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "momentum_rank"
    assert body["status"] == "active"


@pytest.mark.asyncio
async def test_evaluate_returns_buy(wired: tuple[AsyncClient, StrategyService]):
    client, _ = wired
    resp = await client.post("/api/v1/strategy/evaluate/aapl")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "AAPL"
    assert body["signal"] == "BUY"
    assert body["stop_loss"] == 95.0


@pytest.mark.asyncio
async def test_decay_deactivates(wired: tuple[AsyncClient, StrategyService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/strategy/decay",
        json={
            "sharpe_30d": -0.5,
            "sharpe_90d": 0.0,
            "sharpe_180d": 0.0,
            "win_rate_30d": 0.5,
            "profit_factor_30d": 1.0,
            "excess_return_vs_spy_30d": 0.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "deactivated"
    assert body["status_changed"] is True


@pytest.mark.asyncio
async def test_evaluate_feature_engine_unreachable_502(wired_failing: AsyncClient):
    resp = await wired_failing.post("/api/v1/strategy/evaluate/AAPL")
    assert resp.status_code == 502
