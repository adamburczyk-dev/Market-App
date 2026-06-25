"""NATS event publishing for market-data."""

from typing import Protocol

import structlog
from nats.aio.client import Client as NatsClient
from trading_common.events import BaseEvent

logger = structlog.get_logger()


class Publisher(Protocol):
    async def publish(self, event: BaseEvent) -> None: ...


class NullPublisher:
    """No-op publisher. Used in tests and when NATS is unavailable."""

    def __init__(self) -> None:
        self.published: list[BaseEvent] = []

    async def publish(self, event: BaseEvent) -> None:
        self.published.append(event)


class NatsPublisher:
    """Publish events to NATS on their declared subject."""

    def __init__(self, client: NatsClient) -> None:
        self._client = client

    async def publish(self, event: BaseEvent) -> None:
        await self._client.publish(event.subject(), event.model_dump_json().encode())
        logger.info("Published event", subject=event.subject(), event_id=event.event_id)
