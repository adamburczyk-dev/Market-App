from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_service
from src.core.service import DashboardService

_DEFAULTS: dict[str, dict] = {
    "rp": {
        "value": 100000.0,
        "exposure_pct": 0.2,
        "drawdown_pct": 0.03,
        "daily_loss_pct": 0.0,
        "regime": "expansion",
    },
    "cb": {"level": "none", "tripped": False},
    "ep": {"cash": 95000.0, "equity": 100000.0, "exposure_pct": 0.05},
    "pos": {"positions": {"AAPL": {"quantity": 50, "last_price": 100.0}}},
    "al": {"alerts": [{"severity": "critical", "title": "Circuit breaker RED"}]},
    "ml": {"models": ["m1"]},
}


class FakeSource:
    """DashboardSource double — healthy by default; pass key=None to mark a source down."""

    def __init__(self, **overrides: dict | None) -> None:
        # an override (even None) replaces the default → key=None simulates an unavailable upstream
        self._data: dict[str, dict | None] = {**_DEFAULTS, **overrides}

    async def risk_portfolio(self) -> dict | None:
        return self._data["rp"]

    async def circuit_breaker(self) -> dict | None:
        return self._data["cb"]

    async def execution_portfolio(self) -> dict | None:
        return self._data["ep"]

    async def positions(self) -> dict | None:
        return self._data["pos"]

    async def recent_alerts(self) -> dict | None:
        return self._data["al"]

    async def models(self) -> dict | None:
        return self._data["ml"]

    async def aclose(self) -> None:
        return None


def build_service(source: FakeSource | None = None) -> DashboardService:
    return DashboardService(source or FakeSource())


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, DashboardService]]:
    from src.main import app

    service = build_service()
    app.dependency_overrides[get_service] = lambda: service
    app.state.service = service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, service
    finally:
        app.dependency_overrides.clear()
        if hasattr(app.state, "service"):
            delattr(app.state, "service")
