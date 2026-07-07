"""Durable JetStream subscriptions dispatching to async handlers.

Used for the three source subjects feature-engine consumes:
``market_data.updated`` (Tier-1 technical computation), ``fundamentals.updated``
and ``company.classified`` (Tier-2 attribute enrichment). Poison-message safe:
a malformed payload is terminated (no redelivery), a transient failure (e.g.
an upstream temporarily down) is NAK'd and redelivered up to ``max_deliver``
times.
"""

from collections.abc import Awaitable, Callable
from contextlib import suppress

import structlog
from nats.js import JetStreamContext
from nats.js.api import ConsumerConfig
from pydantic import ValidationError

logger = structlog.get_logger()

Handler = Callable[[bytes], Awaitable[None]]


class EventSubscriber:
    """Durable push subscription to a JetStream subject."""

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
        logger.info("Subscribed to events", subject=self._subject, durable=self._durable)

    async def _on_message(self, msg) -> None:  # type: ignore[no-untyped-def]
        try:
            await self._handler(msg.data)
            await msg.ack()
        except (ValidationError, ValueError) as exc:
            # Malformed/undecodable event — redelivery won't help. Drop it.
            logger.error("Poison message, terminating", error=str(exc))
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
