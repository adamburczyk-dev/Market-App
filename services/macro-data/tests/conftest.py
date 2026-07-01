from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_service
from src.core.service import MacroDataService
from src.events.publisher import NullPublisher


class FakeFetcher:
    """MacroFetcher double — returns configured indicator values (default: none)."""

    def __init__(self, indicators: dict | None = None, enabled: bool = False) -> None:
        self._indicators = indicators or {}
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def fetch_indicators(self) -> dict:
        return dict(self._indicators)

    async def aclose(self) -> None:
        return None


def build_service(fetcher: FakeFetcher | None = None, publisher=None):  # type: ignore[no-untyped-def]
    return MacroDataService(fetcher or FakeFetcher(), publisher or NullPublisher())


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, MacroDataService]]:
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
