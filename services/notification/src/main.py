from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import nats
import structlog
from fastapi import FastAPI

from src.api import router as api_router
from src.config import settings
from src.core.channels import Channel, LogChannel, SlackChannel, TelegramChannel
from src.core.observability import setup_observability
from src.core.service import NotificationService
from src.events.subscriber import EventSubscriber, ensure_stream

logger = structlog.get_logger()


def build_channels() -> list[Channel]:
    """LogChannel is always on; Slack/Telegram only when configured."""
    channels: list[Channel] = [LogChannel()]
    if settings.SLACK_WEBHOOK_URL:
        channels.append(SlackChannel(settings.SLACK_WEBHOOK_URL))
    if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
        channels.append(TelegramChannel(settings.TELEGRAM_BOT_TOKEN, settings.TELEGRAM_CHAT_ID))
    return channels


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)

    channels = build_channels()
    service = NotificationService(channels, min_severity=settings.MIN_SEVERITY)
    app.state.service = service
    logger.info("Channels enabled", channels=[ch.name for ch in channels])

    nats_client = None
    subscribers: list[EventSubscriber] = []
    try:
        nats_client = await nats.connect(settings.NATS_URL)
        js = nats_client.jetstream()
        # Ensure every consumed stream exists so subscription is start-order independent.
        await ensure_stream(js, settings.NATS_RISK_STREAM, ["risk.>"])
        await ensure_stream(js, settings.NATS_ORDERS_STREAM, ["order.>"])
        await ensure_stream(js, settings.NATS_BACKTEST_STREAM, ["backtest.>"])
        await ensure_stream(js, settings.NATS_ML_STREAM, ["ml.>"])

        plan = [
            (settings.NATS_RISK_SUBJECT, "notification-risk", service.handle_circuit_breaker),
            (settings.NATS_ORDERS_SUBJECT, "notification-orders", service.handle_order_filled),
            (
                settings.NATS_BACKTEST_SUBJECT,
                "notification-backtest",
                service.handle_strategy_revalidated,
            ),
            (settings.NATS_ML_SUBJECT, "notification-ml", service.handle_model_drift),
        ]
        for subject, durable, handler in plan:
            sub = EventSubscriber(
                nats_client.jetstream(),
                subject,
                durable,
                handler,
                max_deliver=settings.NATS_MAX_DELIVER,
            )
            await sub.start()
            subscribers.append(sub)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS unavailable, alerts disabled", error=str(exc))
        nats_client = None

    async def _readiness() -> tuple[bool, dict[str, bool]]:
        # notification's whole job is consuming events → NATS is required.
        nats_ok = nats_client is not None and nats_client.is_connected
        return nats_ok, {"nats": nats_ok}

    app.state.readiness_check = _readiness

    yield

    logger.info("Shutting down service", service=settings.SERVICE_NAME)
    for sub in subscribers:
        await sub.stop()
    if nats_client is not None:
        with suppress(Exception):
            await nats_client.drain()
    for ch in channels:
        with suppress(Exception):
            await ch.aclose()


app = FastAPI(
    title="Notification Service",
    description="Alerty: Telegram, email, Slack",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
