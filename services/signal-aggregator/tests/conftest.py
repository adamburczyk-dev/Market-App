from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from trading_common.cost_filter import CostAwareFilter

from src.api.deps import get_service
from src.core.adaptive_weights import AdaptiveWeightOptimizer
from src.core.aggregator import SignalComponent
from src.core.service import SignalAggregatorService

SOURCES = ["strategy", "ml", "macro"]


def components(*triples: tuple[str, str, float]) -> list[SignalComponent]:
    return [SignalComponent(*t) for t in triples]


def build_service(publisher=None, sources=None, cost_filter=None, **kwargs):  # type: ignore[no-untyped-def]
    from src.events.publisher import NullPublisher

    optimizer = AdaptiveWeightOptimizer(sources if sources is not None else SOURCES)
    return SignalAggregatorService(
        optimizer,
        cost_filter or CostAwareFilter(),
        publisher or NullPublisher(),
        **kwargs,
    )


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, SignalAggregatorService]]:
    from src.events.publisher import NullPublisher
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
