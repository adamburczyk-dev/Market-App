from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import nats
import structlog
from fastapi import FastAPI

from src.api import router as api_router
from src.config import settings
from src.core.fred_client import FredClient
from src.core.observability import setup_observability
from src.core.service import MacroDataService
from src.events.publisher import NatsPublisher, NullPublisher, Publisher, ensure_stream

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)

    fetcher = FredClient(settings.FRED_API_KEY, base_url=settings.FRED_BASE_URL)
    if not fetcher.enabled:
        logger.warning("FRED_API_KEY not set — relying on manually-posted indicators")

    publisher: Publisher
    nats_client = None
    try:
        nats_client = await nats.connect(settings.NATS_URL)
        js = nats_client.jetstream()
        await ensure_stream(js, settings.NATS_MACRO_STREAM, [settings.NATS_MACRO_SUBJECTS])
        publisher = NatsPublisher(js)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS/JetStream unavailable, events disabled", error=str(exc))
        publisher = NullPublisher()
        nats_client = None

    service = MacroDataService(fetcher, publisher)
    app.state.service = service

    async def _readiness() -> tuple[bool, dict[str, bool]]:
        # macro-data publishes regime events → NATS is required.
        nats_ok = nats_client is not None and nats_client.is_connected
        return nats_ok, {"nats": nats_ok}

    app.state.readiness_check = _readiness

    yield

    logger.info("Shutting down service", service=settings.SERVICE_NAME)
    if nats_client is not None:
        with suppress(Exception):
            await nats_client.drain()
    await fetcher.aclose()


app = FastAPI(
    title="Macro Data Service",
    description="FRED macro indicators + market regime detection",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
