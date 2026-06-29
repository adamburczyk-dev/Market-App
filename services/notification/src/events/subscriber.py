"""Durable JetStream push subscription that dispatches messages to a handler.

Poison-message safe: malformed payloads are terminated; transient failures are
NAK'd and redelivered up to ``max_deliver`` times. notification runs one per
subscribed subject (circuit breaker, fills, revalidation, drift).
"""

from collections.abc import Awaitable, Callable
from contextlib import suppress

import structlog
from nats.js import JetStreamContext
from nats.js.api import ConsumerConfig
from nats.js.errors import NotFoundError
from pydantic import ValidationError

logger = structlog.get_logger()

Handler = Callable[[bytes], Awaitable[None]]


async def ensure_stream(js: JetStreamContext, name: str, subjects: list[str]) -> None:
    """Create the JetStream stream if it does not already exist (idempotent).

    notification consumes streams owned by other services; ensuring them here makes
    subscription independent of start order.
    """
    try:
        await js.stream_info(name)
    except NotFoundError:
        await js.add_stream(name=name, subjects=subjects)
        logger.info("Created JetStream stream", stream=name, subjects=subjects)


class EventSubscriber:
    def __init__(
        self,
        js: JetStreamContext,
        subject: str,
        durable: str,
        handler: Handler,
        max_deliver: int = 5,
    ) -> None:
        self._js = js
        self._subject = subject
        self._durable = durable
        self._handler = handler
        self._max_deliver = max_deliver
        self._sub: JetStreamContext.PushSubscription | None = None

    async def start(self) -> None:
        self._sub = await self._js.subscribe(
            self._subject,
            durable=self._durable,
            cb=self._on_message,
            manual_ack=True,
            config=ConsumerConfig(max_deliver=self._max_deliver),
        )
        logger.info("Subscribed", subject=self._subject, durable=self._durable)

    async def _on_message(self, msg) -> None:  # type: ignore[no-untyped-def]
        try:
            await self._handler(msg.data)
            await msg.ack()
        except (ValidationError, ValueError) as exc:
            logger.error("Poison message, terminating", subject=self._subject, error=str(exc))
            with suppress(Exception):
                await msg.term()
        except Exception as exc:  # noqa: BLE001 - transient; redeliver up to max_deliver
            logger.warning("Transient error, will redeliver", error=str(exc))
            with suppress(Exception):
                await msg.nak()

    async def stop(self) -> None:
        if self._sub is not None:
            with suppress(Exception):
                await self._sub.unsubscribe()
