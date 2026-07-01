"""Tests for signal-aggregator HTTP routes."""

import pytest
from httpx import AsyncClient

from src.core.service import SignalAggregatorService


def agg_body(**over: object) -> dict:
    body: dict = {
        "symbol": "AAPL",
        "components": [
            {"source": "strategy", "signal": "BUY", "confidence": 0.9},
            {"source": "ml", "signal": "BUY", "confidence": 0.8},
            {"source": "macro", "signal": "BUY", "confidence": 0.7},
        ],
    }
    body.update(over)
    return body


@pytest.mark.asyncio
async def test_status_ok(client: AsyncClient):
    resp = await client.get("/api/v1/signal-aggregator/status")
    assert resp.status_code == 200
    assert resp.json()["service"] == "signal-aggregator"


@pytest.mark.asyncio
async def test_aggregate_endpoint(wired: tuple[AsyncClient, SignalAggregatorService]):
    client, _ = wired
    resp = await client.post("/api/v1/signal-aggregator/aggregate", json=agg_body())
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_signal"] == "BUY"
    assert body["components_count"] == 3
    assert set(body["weights"]) == {"strategy", "ml", "macro"}


@pytest.mark.asyncio
async def test_aggregate_cost_filtered(wired: tuple[AsyncClient, SignalAggregatorService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/signal-aggregator/aggregate",
        json=agg_body(expected_return_bps=5.0, market_cap_tier="micro"),
    )
    body = resp.json()
    assert body["final_signal"] == "HOLD"
    assert body["cost_filtered"] is True


@pytest.mark.asyncio
async def test_weights_endpoint(wired: tuple[AsyncClient, SignalAggregatorService]):
    client, _ = wired
    resp = await client.get("/api/v1/signal-aggregator/weights")
    assert resp.status_code == 200
    assert set(resp.json()["weights"]) == {"strategy", "ml", "macro"}


@pytest.mark.asyncio
async def test_outcomes_endpoint(wired: tuple[AsyncClient, SignalAggregatorService]):
    client, _ = wired
    resp = await client.post(
        "/api/v1/signal-aggregator/outcomes", json={"source": "ml", "daily_return": 0.02}
    )
    assert resp.status_code == 200
    assert resp.json()["recorded"] is True


@pytest.mark.asyncio
async def test_invalid_confidence_rejected(wired: tuple[AsyncClient, SignalAggregatorService]):
    client, _ = wired
    bad = agg_body()
    bad["components"][0]["confidence"] = 1.5  # > 1.0
    resp = await client.post("/api/v1/signal-aggregator/aggregate", json=bad)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_aggregate_503_when_unwired(client: AsyncClient):
    resp = await client.post("/api/v1/signal-aggregator/aggregate", json=agg_body())
    assert resp.status_code == 503
