"""Tests for notification HTTP routes."""

import pytest
from httpx import AsyncClient

from src.core.service import NotificationService


@pytest.mark.asyncio
async def test_status_ok(client: AsyncClient):
    resp = await client.get("/api/v1/notification/status")
    assert resp.status_code == 200
    assert resp.json()["service"] == "notification"


@pytest.mark.asyncio
async def test_channels_lists_collecting(wired: tuple[AsyncClient, NotificationService]):
    client, _ = wired
    resp = await client.get("/api/v1/notification/channels")
    assert resp.status_code == 200
    assert resp.json()["channels"] == ["collect"]


@pytest.mark.asyncio
async def test_test_alert_dispatches(wired: tuple[AsyncClient, NotificationService]):
    client, service = wired
    resp = await client.post(
        "/api/v1/notification/test-alert",
        json={"severity": "warning", "title": "Hi", "message": "there"},
    )
    assert resp.status_code == 200
    assert resp.json()["dispatched"] is True
    recent = service.recent()
    assert recent[-1].title == "Hi"


@pytest.mark.asyncio
async def test_recent_alerts_endpoint(wired: tuple[AsyncClient, NotificationService]):
    client, _ = wired
    await client.post("/api/v1/notification/test-alert", json={"title": "A1"})
    resp = await client.get("/api/v1/notification/alerts/recent")
    assert resp.status_code == 200
    titles = [a["title"] for a in resp.json()["alerts"]]
    assert "A1" in titles


@pytest.mark.asyncio
async def test_channels_503_when_unwired(client: AsyncClient):
    resp = await client.get("/api/v1/notification/channels")
    assert resp.status_code == 503
