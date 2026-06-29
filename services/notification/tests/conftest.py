from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_service
from src.core.channels import Alert
from src.core.service import NotificationService


class CollectingChannel:
    """Test channel double — records dispatched alerts."""

    name = "collect"

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    async def send(self, alert: Alert) -> None:
        self.sent.append(alert)

    async def aclose(self) -> None:
        return None


class FailingChannel:
    name = "failing"

    async def send(self, alert: Alert) -> None:
        raise RuntimeError("delivery boom")

    async def aclose(self) -> None:
        return None


def build_service(channels=None, min_severity="info"):  # type: ignore[no-untyped-def]
    return NotificationService(
        channels if channels is not None else [CollectingChannel()],
        min_severity=min_severity,
    )


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def wired() -> AsyncIterator[tuple[AsyncClient, NotificationService]]:
    from src.main import app

    service = build_service([CollectingChannel()])
    app.dependency_overrides[get_service] = lambda: service
    app.state.service = service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, service
    finally:
        app.dependency_overrides.clear()
        if hasattr(app.state, "service"):
            delattr(app.state, "service")
