from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime

import nats
import structlog
from fastapi import FastAPI
from trading_common.scheduler import SECONDS_PER_WEEK, PeriodicTask, seconds_until_weekday_hour
from trading_common.schemas import Interval

from src.api import router as api_router
from src.config import settings
from src.core.engine import BacktestParams
from src.core.market_data_client import HttpMarketDataClient
from src.core.observability import setup_observability
from src.core.service import BacktestService
from src.events.publisher import NatsPublisher, NullPublisher, Publisher, ensure_stream

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)

    market_client = HttpMarketDataClient(settings.MARKET_DATA_URL)

    publisher: Publisher
    nats_client = None
    try:
        nats_client = await nats.connect(settings.NATS_URL)
        js = nats_client.jetstream()
        await ensure_stream(js, settings.NATS_BACKTEST_STREAM, [settings.NATS_BACKTEST_SUBJECTS])
        publisher = NatsPublisher(js)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS/JetStream unavailable, events disabled", error=str(exc))
        publisher = NullPublisher()
        nats_client = None

    service = BacktestService(
        market_client,
        publisher,
        default_params=BacktestParams(
            lookback=settings.BACKTEST_LOOKBACK,
            entry_momentum=settings.BACKTEST_ENTRY_MOMENTUM,
            cost_bps=settings.BACKTEST_COST_BPS,
        ),
        oos_window_days=settings.OOS_WINDOW_DAYS,
        is_window_days=settings.IS_WINDOW_DAYS,
        degradation_threshold=settings.DEGRADATION_THRESHOLD,
    )
    app.state.service = service

    scheduler: PeriodicTask | None = None
    if settings.SCHEDULE_REVALIDATION_ENABLED:

        async def _weekly_revalidation() -> None:
            await service.revalidate(
                settings.REVALIDATION_STRATEGY,
                settings.REVALIDATION_SYMBOL,
                settings.REVALIDATION_ORIGINAL_OOS_SHARPE,
                Interval(settings.REVALIDATION_INTERVAL),
                limit=settings.BACKTEST_DEFAULT_LIMIT,
            )

        scheduler = PeriodicTask(
            "weekly-revalidation",
            interval_s=SECONDS_PER_WEEK,
            job=_weekly_revalidation,
            initial_delay_s=seconds_until_weekday_hour(
                datetime.now(UTC),
                settings.REVALIDATION_WEEKDAY,
                settings.REVALIDATION_HOUR_UTC,
            ),
        )
        scheduler.start()

    async def _readiness() -> tuple[bool, dict[str, bool]]:
        # backtest is request-driven; its hard dependency is market-data (historical OHLCV).
        market_ok = await market_client.health_ok()
        return market_ok, {"market_data": market_ok}

    app.state.readiness_check = _readiness

    yield

    logger.info("Shutting down service", service=settings.SERVICE_NAME)
    if scheduler is not None:
        await scheduler.stop()
    if nats_client is not None:
        with suppress(Exception):
            await nats_client.drain()
    await market_client.aclose()


app = FastAPI(
    title="Backtest Service",
    description="Silnik backtestingu i walk-forward rewalidacja strategii",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
