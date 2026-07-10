from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import nats
import structlog
from fastapi import FastAPI
from trading_common.scheduler import PeriodicTask

from src.api import router as api_router
from src.config import settings
from src.core.edgar_client import EdgarClient
from src.core.observability import setup_observability
from src.core.service import FundamentalDataService
from src.events.publisher import NatsPublisher, NullPublisher, Publisher, ensure_stream

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)

    fetcher = EdgarClient(
        settings.SEC_USER_AGENT,
        base_url=settings.SEC_BASE_URL,
        tickers_url=settings.SEC_TICKERS_URL,
    )
    if not fetcher.enabled:
        logger.warning("SEC_USER_AGENT not set — relying on manually-posted statements")

    publisher: Publisher
    nats_client = None
    try:
        nats_client = await nats.connect(settings.NATS_URL)
        js = nats_client.jetstream()
        await ensure_stream(
            js, settings.NATS_FUNDAMENTALS_STREAM, [settings.NATS_FUNDAMENTALS_SUBJECTS]
        )
        publisher = NatsPublisher(js)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS/JetStream unavailable, events disabled", error=str(exc))
        publisher = NullPublisher()
        nats_client = None

    service = FundamentalDataService(fetcher, publisher)
    app.state.service = service

    scheduler: PeriodicTask | None = None
    refresh_symbols = settings.refresh_symbols
    if settings.SCHEDULE_REFRESH_ENABLED and fetcher.enabled and refresh_symbols:

        async def _refresh_job() -> None:
            await service.refresh_universe(refresh_symbols, pause_s=settings.REFRESH_SYMBOL_PAUSE_S)

        scheduler = PeriodicTask(
            "fundamentals-refresh",
            interval_s=settings.REFRESH_INTERVAL_S,
            job=_refresh_job,
            initial_delay_s=settings.REFRESH_INITIAL_DELAY_S,
        )
        scheduler.start()
    elif settings.SCHEDULE_REFRESH_ENABLED:
        logger.info(
            "Scheduled fundamentals refresh skipped — needs SEC_USER_AGENT and REFRESH_SYMBOLS"
        )

    async def _readiness() -> tuple[bool, dict[str, bool]]:
        # fundamental-data publishes fundamentals events → NATS is required.
        nats_ok = nats_client is not None and nats_client.is_connected
        return nats_ok, {"nats": nats_ok}

    app.state.readiness_check = _readiness

    yield

    logger.info("Shutting down service", service=settings.SERVICE_NAME)
    if scheduler is not None:
        await scheduler.stop()
    if nats_client is not None:
        with suppress(Exception):
            await nats_client.drain()
    await fetcher.aclose()


app = FastAPI(
    title="Fundamental Data Service",
    description="SEC EDGAR fundamentals + Piotroski F-Score",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
