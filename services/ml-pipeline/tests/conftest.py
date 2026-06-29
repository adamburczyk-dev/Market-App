from collections.abc import AsyncIterator

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_service
from src.core.monitoring.drift_detector import DriftDetector
from src.core.registry import ModelRegistry
from src.core.service import MLPipelineService
from src.events.publisher import NullPublisher


def normal_samples(mean: float, std: float, n: int = 500, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.normal(mean, std, n)]


def build_service(publisher=None, registry=None):  # type: ignore[no-untyped-def]
    return MLPipelineService(
        DriftDetector(),
        registry or ModelRegistry(),
        publisher or NullPublisher(),
    )


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, MLPipelineService]]:
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
