from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import nats
import structlog
from fastapi import FastAPI

from src.api import router as api_router
from src.config import settings
from src.core.market_data_client import HttpMarketDataClient
from src.core.observability import setup_observability
from src.core.service import FeatureEngineService
from src.core.store import FeatureStore
from src.events.publisher import NatsPublisher, NullPublisher, Publisher, ensure_stream
from src.events.subscriber import MarketDataSubscriber

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)

    client = HttpMarketDataClient(settings.MARKET_DATA_URL)
    store = FeatureStore()

    publisher: Publisher
    nats_client = None
    subscriber: MarketDataSubscriber | None = None
    try:
        nats_client = await nats.connect(settings.NATS_URL)
        js = nats_client.jetstream()
        await ensure_stream(js, settings.NATS_FEATURES_STREAM, [settings.NATS_FEATURES_SUBJECTS])
        # Ensure the source stream too, so subscription works regardless of start order.
        await ensure_stream(js, settings.NATS_SOURCE_STREAM, ["market_data.>"])
        publisher = NatsPublisher(js)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS/JetStream unavailable, events disabled", error=str(exc))
        publisher = NullPublisher()
        nats_client = None

    service = FeatureEngineService(
        client,
        store,
        publisher,
        lookback=settings.FEATURE_LOOKBACK,
        min_bars=settings.FEATURE_MIN_BARS,
    )
    app.state.service = service

    if nats_client is not None:
        try:
            subscriber = MarketDataSubscriber(
                nats_client.jetstream(),
                settings.NATS_SOURCE_SUBJECT,
                settings.NATS_DURABLE,
                service.handle_market_data_event,
                max_deliver=settings.NATS_MAX_DELIVER,
            )
            await subscriber.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not subscribe to market-data events", error=str(exc))
            subscriber = None

    yield

    logger.info("Shutting down service", service=settings.SERVICE_NAME)
    if subscriber is not None:
        await subscriber.stop()
    if nats_client is not None:
        with suppress(Exception):
            await nats_client.drain()
    await client.aclose()


app = FastAPI(
    title="Feature Engine Service",
    description="Obliczanie wskaźników technicznych i feature engineeringu",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
