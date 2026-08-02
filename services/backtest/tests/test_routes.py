"""Tests for backtest HTTP routes (service wired via dependency override)."""

import pytest
from httpx import AsyncClient

from src.core.service import BacktestService


@pytest.mark.asyncio
async def test_status_ok(client: AsyncClient):
    resp = await client.get("/api/v1/backtest/status")
    assert resp.status_code == 200
    assert resp.json()["service"] == "backtest"


@pytest.mark.asyncio
async def test_run_endpoint_returns_metrics(wired: tuple[AsyncClient, BacktestService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/backtest/run",
        json={"strategy_name": "sma_ema_crossover", "symbol": "AAPL", "interval": "1d"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy_name"] == "sma_ema_crossover"
    assert "sharpe_ratio" in body
    assert "total_return" in body
    assert body["n_bars"] > 0


@pytest.mark.asyncio
async def test_revalidate_endpoint_returns_recommendation(
    wired: tuple[AsyncClient, BacktestService],
):
    client, _ = wired
    resp = await client.post(
        "/api/v1/backtest/revalidate",
        json={
            "strategy_name": "sma_ema_crossover",
            "symbol": "AAPL",
            "original_oos_sharpe": 1.0,
            "interval": "1d",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommended_status"] in {"active", "probation", "deactivate"}
    assert body["oos_window_days"] == 126


@pytest.mark.asyncio
async def test_run_returns_503_when_service_unwired(client: AsyncClient):
    # plain client fixture does not run lifespan and has no dependency override
    resp = await client.post(
        "/api/v1/backtest/run",
        json={"strategy_name": "sma_ema_crossover", "symbol": "AAPL", "interval": "1d"},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_run_validation_rejects_bad_limit(wired: tuple[AsyncClient, BacktestService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/backtest/run",
        json={"strategy_name": "sma_ema_crossover", "symbol": "AAPL", "interval": "1d", "limit": 1},
    )
    assert resp.status_code == 422  # limit below ge=10


@pytest.mark.asyncio
async def test_unknown_strategy_is_404_naming_the_alternatives(
    wired: tuple[AsyncClient, BacktestService],
):
    client, _ = wired
    resp = await client.post(
        "/api/v1/backtest/run",
        json={"strategy_name": "nie_ma_takiej", "symbol": "AAPL", "interval": "1d"},
    )
    assert resp.status_code == 404
    assert "sma_ema_crossover" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_cross_sectional_strategy_is_422_saying_why(
    wired: tuple[AsyncClient, BacktestService],
):
    """Not a 500 and not a silent proxy: the caller has to learn that this rule
    needs a universe."""
    client, _ = wired
    resp = await client.post(
        "/api/v1/backtest/run",
        json={"strategy_name": "momentum_rank", "symbol": "AAPL", "interval": "1d"},
    )
    assert resp.status_code == 422
    assert "momentum_20" in resp.json()["detail"]
