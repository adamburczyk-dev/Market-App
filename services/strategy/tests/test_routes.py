"""Testy HTTP endpointów strategy."""

import pytest
from httpx import AsyncClient

from src.core.service import StrategyService

DECAY_METRICS = {
    "sharpe_30d": -0.5,
    "sharpe_90d": 0.0,
    "sharpe_180d": 0.0,
    "win_rate_30d": 0.5,
    "profit_factor_30d": 1.0,
    "excess_return_vs_spy_30d": 0.0,
}


@pytest.mark.asyncio
async def test_status_lists_every_rule(wired: tuple[AsyncClient, StrategyService]):
    client, _ = wired
    resp = await client.get("/api/v1/strategy/status")
    assert resp.status_code == 200
    strategies = resp.json()["strategies"]
    assert [s["name"] for s in strategies] == ["momentum_rank"]
    assert strategies[0]["status"] == "active"
    assert strategies[0]["required_features"] == ["rsi_14"]
    # The rank is reported apart: it is what makes this rule un-backtestable
    # one symbol at a time.
    assert strategies[0]["required_ranks"] == ["momentum_20"]


@pytest.mark.asyncio
async def test_evaluate_returns_a_signal_per_rule(wired: tuple[AsyncClient, StrategyService]):
    client, _ = wired
    resp = await client.post("/api/v1/strategy/evaluate/aapl")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "AAPL"
    assert len(body["signals"]) == 1
    assert body["signals"][0]["strategy"] == "momentum_rank"
    assert body["signals"][0]["signal"] == "BUY"
    assert body["signals"][0]["stop_loss"] == 95.0


@pytest.mark.asyncio
async def test_decay_deactivates_the_named_strategy(wired: tuple[AsyncClient, StrategyService]):
    client, _ = wired
    resp = await client.post("/api/v1/strategy/decay/momentum_rank", json=DECAY_METRICS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "momentum_rank"
    assert body["status"] == "deactivated"
    assert body["status_changed"] is True


@pytest.mark.asyncio
async def test_decay_for_a_strategy_we_do_not_run_is_404(
    wired: tuple[AsyncClient, StrategyService],
):
    """Not a silent no-op: a metrics push aimed at the wrong name must say so,
    or a strategy would look supervised while nothing was watching it."""
    client, _ = wired
    resp = await client.post("/api/v1/strategy/decay/donchian_breakout", json=DECAY_METRICS)
    assert resp.status_code == 404
    assert "momentum_rank" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_evaluate_feature_engine_unreachable_502(wired_failing: AsyncClient):
    resp = await wired_failing.post("/api/v1/strategy/evaluate/AAPL")
    assert resp.status_code == 502
