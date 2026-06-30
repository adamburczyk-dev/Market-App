"""Tests for HttpDashboardSource — graceful per-call degradation."""

import httpx
import pytest

from src.core.clients import HttpDashboardSource


def source_with(handler) -> HttpDashboardSource:  # type: ignore[no-untyped-def]
    src = HttpDashboardSource(
        risk_url="http://risk:8000",
        execution_url="http://exec:8000",
        notification_url="http://notif:8000",
        ml_url="http://ml:8000",
    )
    # swap in a mock transport
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return src


@pytest.mark.asyncio
async def test_returns_json_on_success():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"value": 42})

    src = source_with(handler)
    result = await src.risk_portfolio()
    await src.aclose()
    assert result == {"value": 42}
    assert captured["url"] == "http://risk:8000/api/v1/risk-mgmt/portfolio"


@pytest.mark.asyncio
async def test_returns_none_on_http_error():
    src = source_with(lambda r: httpx.Response(500))
    assert await src.models() is None
    await src.aclose()


@pytest.mark.asyncio
async def test_returns_none_on_connect_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    src = source_with(boom)
    assert await src.recent_alerts() is None
    await src.aclose()


@pytest.mark.asyncio
async def test_each_endpoint_targets_correct_url():
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={})

    src = source_with(handler)
    await src.risk_portfolio()
    await src.circuit_breaker()
    await src.execution_portfolio()
    await src.positions()
    await src.recent_alerts()
    await src.models()
    await src.aclose()
    assert urls == [
        "http://risk:8000/api/v1/risk-mgmt/portfolio",
        "http://risk:8000/api/v1/risk-mgmt/circuit-breaker",
        "http://exec:8000/api/v1/execution/portfolio",
        "http://exec:8000/api/v1/execution/positions",
        "http://notif:8000/api/v1/notification/alerts/recent",
        "http://ml:8000/api/v1/ml-pipeline/models",
    ]
