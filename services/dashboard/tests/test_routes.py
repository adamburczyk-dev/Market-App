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


@pytest.mark.asyncio
async def test_root_redirects_to_ui():
    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
        assert resp.status_code in (307, 308)
        assert resp.headers["location"] == "/api/v1/dashboard/ui"


@pytest.mark.asyncio
async def test_every_path_the_ui_fetches_is_a_real_route(client: AsyncClient):
    """The strongest thing this page can be tested for: a typo in a fetch path
    renders an empty card and nothing errors, so only a check like this catches
    it. Pins the routes the JS names against the routes the app declares."""
    import re

    from src.main import app

    html = (await client.get("/api/v1/dashboard/ui")).text
    fetched = set(re.findall(r'fetch\("([a-z/]+)"', html)) | set(
        re.findall(r'"(sections/[a-z]+)"', html)
    )
    assert fetched, "the page fetches nothing — the check would pass vacuously"

    # OpenAPI paths, not app.routes: an included router is one entry there and
    # its children would never be seen, so the check would pass vacuously.
    declared = set(app.openapi()["paths"])
    for path in fetched:
        full = f"/api/v1/dashboard/{path}"
        assert full in declared, f"UI fetches {full}, which no route serves"


@pytest.mark.asyncio
async def test_ui_escapes_upstream_text():
    """Alert titles and strategy names come from other services; rendering them
    with innerHTML without escaping would make any upstream a script source."""
    from src.api.ui import INDEX_HTML

    assert "const esc = s => String(s).replace" in INDEX_HTML
    # the renderers of upstream-controlled text must route through it
    assert "esc(s.name)" in INDEX_HTML
    assert "esc(a.title)" in INDEX_HTML or "recent_alerts" not in INDEX_HTML


@pytest.mark.asyncio
async def test_section_endpoints_answer(wired: tuple[AsyncClient, DashboardService]):
    client, _ = wired
    for name in ("portfolio", "risk", "strategy", "ml", "health"):
        resp = await client.get(f"/api/v1/dashboard/sections/{name}")
        assert resp.status_code == 200, name
        assert isinstance(resp.json(), dict)


@pytest.mark.asyncio
async def test_backtest_section_passes_the_upstream_status_through(
    wired: tuple[AsyncClient, DashboardService],
):
    """404 and 422 are answers, not outages — flattening them into 200 would
    report a working service as a broken one."""
    client, _ = wired
    ok = await client.post(
        "/api/v1/dashboard/sections/backtest?strategy=sma_ema_crossover&symbol=aapl"
    )
    assert ok.status_code == 200
    assert ok.json()["symbol"] == "AAPL"  # normalized upstream

    unknown = await client.post("/api/v1/dashboard/sections/backtest?strategy=nie_ma&symbol=AAPL")
    assert unknown.status_code == 404

    cross = await client.post(
        "/api/v1/dashboard/sections/backtest?strategy=momentum_rank&symbol=AAPL"
    )
    assert cross.status_code == 422
    assert "momentum_20" in cross.json()["detail"]
