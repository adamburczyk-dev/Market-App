"""NATS JetStream event publishing for feature-engine."""

from typing import Protocol

import structlog
from nats.js import JetStreamContext
from nats.js.errors import NotFoundError
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
    """Publish events to NATS JetStream with Nats-Msg-Id dedup."""

    def __init__(self, js: JetStreamContext) -> None:
        self._js = js

    async def publish(self, event: BaseEvent) -> None:
        ack = await self._js.publish(
            event.subject(),
            event.model_dump_json().encode(),
            headers={"Nats-Msg-Id": event.event_id},
        )
        logger.info(
            "Published event",
            subject=event.subject(),
            event_id=event.event_id,
            stream=ack.stream,
            seq=ack.seq,
        )


async def ensure_stream(js: JetStreamContext, name: str, subjects: list[str]) -> None:
    """Create the JetStream stream if it does not already exist (idempotent)."""
    try:
        await js.stream_info(name)
    except NotFoundError:
        await js.add_stream(name=name, subjects=subjects)
        logger.info("Created JetStream stream", stream=name, subjects=subjects)
