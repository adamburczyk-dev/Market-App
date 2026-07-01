from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from trading_common.schemas import CompanyProfile

from src.api.deps import get_service
from src.core.service import CompanyClassifierService
from src.events.publisher import NullPublisher


def profile(
    symbol: str = "AAPL",
    sector: str | None = "Information Technology",
    market_cap: float | None = 3e12,
) -> CompanyProfile:
    return CompanyProfile(symbol=symbol, sector=sector, market_cap=market_cap)


def build_service(publisher=None):  # type: ignore[no-untyped-def]
    return CompanyClassifierService(publisher or NullPublisher())


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, CompanyClassifierService]]:
    from src.main import app

    service = build_service(publisher=NullPublisher())
    app.dependency_overrides[get_service] = lambda: service
    app.state.service = service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, service
    finally:
        app.dependency_overrides.clear()
        if hasattr(app.state, "service"):
            delattr(app.state, "service")
