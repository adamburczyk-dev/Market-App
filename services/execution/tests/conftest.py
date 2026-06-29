"""Pytest fixtures dla execution service."""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_service
from src.core.paper_broker import PaperBroker
from src.core.risk_client import NullRiskClient
from src.core.service import ExecutionService
from src.events.publisher import NullPublisher


def build_service(  # type: ignore[no-untyped-def]
    publisher=None, risk_client=None, broker=None, repository=None
):
    return ExecutionService(
        broker or PaperBroker(),
        publisher or NullPublisher(),
        risk_client or NullRiskClient(),
        repository=repository,
    )


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, ExecutionService]]:
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
