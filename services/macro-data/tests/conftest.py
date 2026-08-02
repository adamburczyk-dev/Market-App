from collections.abc import AsyncIterator
from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient
from trading_common.schemas import MacroObservation

from src.api.deps import get_service
from src.core.repository import SqlMacroStore
from src.core.service import MacroDataService
from src.events.publisher import NullPublisher
from src.models.db import Base, make_engine, make_sessionmaker


class FakeFetcher:
    """MacroFetcher double — returns configured indicator values (default: none)."""

    def __init__(
        self,
        indicators: dict | None = None,
        enabled: bool = False,
        vintage: dict[str, list[MacroObservation]] | None = None,
    ) -> None:
        self._indicators = indicators or {}
        self._enabled = enabled
        self.vintage = vintage or {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def fetch_indicators(self) -> dict:
        return dict(self._indicators)

    async def fetch_vintage_history(self, series_id, start=None, end=None):  # type: ignore[no-untyped-def]
        return list(self.vintage.get(series_id, []))

    async def aclose(self) -> None:
        return None


def build_service(fetcher: FakeFetcher | None = None, publisher=None, store=None):  # type: ignore[no-untyped-def]
    return MacroDataService(fetcher or FakeFetcher(), publisher or NullPublisher(), store=store)


def obs(series: str, observed: str, value: float, vintage: str | None) -> MacroObservation:
    """Shorthand: one reading with both of its dates."""
    return MacroObservation(
        series=series,
        observation_date=date.fromisoformat(observed),
        value=value,
        realtime_start=date.fromisoformat(vintage) if vintage else None,
    )


@pytest.fixture
async def store() -> AsyncIterator[SqlMacroStore]:
    """A real SqlMacroStore on in-memory sqlite — the queries are exercised, not faked."""
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield SqlMacroStore(make_sessionmaker(engine))
    await engine.dispose()


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
