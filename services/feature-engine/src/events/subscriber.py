"""Subscribe to MarketDataUpdatedEvent from JetStream and dispatch to a handler."""

from collections.abc import Awaitable, Callable
from contextlib import suppress

import structlog
from nats.js import JetStreamContext

logger = structlog.get_logger()

Handler = Callable[[bytes], Awaitable[None]]


class MarketDataSubscriber:
    """Durable push subscription to a market-data subject."""

    def __init__(
        self,
        js: JetStreamContext,
        subject: str,
        durable: str,
        handler: Handler,
    ) -> None:
        self._js = js
        self._subject = subject
        self._durable = durable
        self._handler = handler
        self._sub: JetStreamContext.PushSubscription | None = None

    async def start(self) -> None:
        self._sub = await self._js.subscribe(
            self._subject,
            durable=self._durable,
            cb=self._on_message,
            manual_ack=True,
        )
        logger.info(
            "Subscribed to market-data events", subject=self._subject, durable=self._durable
        )

    async def _on_message(self, msg) -> None:  # type: ignore[no-untyped-def]
        try:
            await self._handler(msg.data)
            await msg.ack()
        except Exception as exc:  # noqa: BLE001 - log and leave unacked for redelivery
            logger.error("Failed to handle market-data event", error=str(exc))

    async def stop(self) -> None:
        if self._sub is not None:
            with suppress(Exception):
                await self._sub.unsubscribe()
