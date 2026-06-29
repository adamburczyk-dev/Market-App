"""Pytest fixtures dla risk-mgmt service."""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_service
from src.core.circuit_breaker import CircuitBreaker
from src.core.portfolio import PortfolioState
from src.core.service import RiskMgmtService
from src.core.sizing import PositionSizer
from src.events.publisher import NullPublisher


def build_service(publisher=None, portfolio=None, repository=None):  # type: ignore[no-untyped-def]
    return RiskMgmtService(
        publisher or NullPublisher(),
        PositionSizer(),
        CircuitBreaker(),
        portfolio or PortfolioState(),
        repository,
    )


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, RiskMgmtService]]:
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
