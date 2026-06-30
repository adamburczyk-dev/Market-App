"""Tests for dashboard HTTP routes."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.service import DashboardService


@pytest.mark.asyncio
async def test_status_ok(client: AsyncClient):
    resp = await client.get("/api/v1/dashboard/status")
    assert resp.status_code == 200
    assert resp.json()["service"] == "dashboard"


@pytest.mark.asyncio
async def test_overview_endpoint(wired: tuple[AsyncClient, DashboardService]):
    client, _ = wired
    resp = await client.get("/api/v1/dashboard/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"]["risk-mgmt"] == "ok"
    assert "portfolio" in body
    assert "recent_alerts" in body


@pytest.mark.asyncio
async def test_overview_503_when_unwired(client: AsyncClient):
    resp = await client.get("/api/v1/dashboard/overview")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_ui_serves_html(client: AsyncClient):
    resp = await client.get("/api/v1/dashboard/ui")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Trading System" in resp.text
    assert "overview" in resp.text  # JS fetches the overview endpoint


@pytest.mark.asyncio
async def test_root_redirects_to_ui():
    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
        assert resp.status_code in (307, 308)
        assert resp.headers["location"] == "/api/v1/dashboard/ui"
