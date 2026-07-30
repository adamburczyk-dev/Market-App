from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime

import nats
import redis.asyncio as aredis
import structlog
from fastapi import FastAPI
from sqlalchemy import text
from trading_common.scheduler import PeriodicTask, seconds_until_hour
from trading_common.schemas import Interval

from src.api import router as api_router
from src.config import settings
from src.core.cache import Cache, InMemoryCache, RedisCache
from src.core.fetchers import build_default_fetcher
from src.core.incremental import is_weekend
from src.core.observability import setup_observability
from src.core.service import MarketDataService
from src.core.storage import OHLCVRepository
from src.events.publisher import NatsPublisher, NullPublisher, Publisher, ensure_stream
from src.models.db import Base, make_engine, make_sessionmaker

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME, log_level=settings.LOG_LEVEL)

    engine = make_engine(settings.database_url)
    sessionmaker = make_sessionmaker(engine)
    # Ensure the table exists. A no-op against the pre-created TimescaleDB hypertable.
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # `create_all` creates missing TABLES, never missing COLUMNS — a
            # database created before adj_close existed would keep silently
            # storing bars without it. No migration tool in this project, so
            # the one additive, idempotent statement lives here. Postgres only:
            # a fresh sqlite (tests) already gets the column from create_all.
            if engine.dialect.name == "postgresql":
                await conn.execute(
                    text("ALTER TABLE ohlcv ADD COLUMN IF NOT EXISTS adj_close DOUBLE PRECISION")
                )
    except Exception as exc:  # noqa: BLE001 - keep the app up for health probes
        logger.error("Database init failed", error=str(exc))
    repository = OHLCVRepository(sessionmaker)

    cache: Cache
    redis_client: aredis.Redis | None = None
    try:
        redis_client = aredis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        cache = RedisCache(redis_client, settings.CACHE_TTL_SECONDS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis unavailable, using in-memory cache", error=str(exc))
        cache = InMemoryCache()
        redis_client = None

    publisher: Publisher
    nats_client = None
    try:
        nats_client = await nats.connect(settings.NATS_URL)
        js = nats_client.jetstream()
        await ensure_stream(js, settings.NATS_STREAM_NAME, [settings.NATS_STREAM_SUBJECTS])
        publisher = NatsPublisher(js)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS/JetStream unavailable, event publishing disabled", error=str(exc))
        publisher = NullPublisher()
        nats_client = None

    fetcher = build_default_fetcher(settings)
    service = MarketDataService(fetcher, repository, cache, publisher)
    app.state.service = service

    scheduler: PeriodicTask | None = None
    fetch_symbols = settings.fetch_symbols
    if settings.SCHEDULE_FETCH_ENABLED and fetch_symbols:

        async def _sync_job() -> None:
            now = datetime.now(UTC)
            if settings.FETCH_SKIP_WEEKENDS and is_weekend(now):
                logger.info("Scheduled pull skipped — weekend", at=now.isoformat())
                return
            await service.sync_universe(
                fetch_symbols,
                Interval(settings.DEFAULT_FETCH_INTERVAL),
                pause_s=settings.FETCH_SYMBOL_PAUSE_S,
                now=now,
                overlap_days=settings.FETCH_OVERLAP_DAYS,
                initial_history_days=settings.FETCH_INITIAL_HISTORY_DAYS,
            )

        scheduler = PeriodicTask(
            "market-data-sync",
            interval_s=settings.FETCH_INTERVAL_S,
            job=_sync_job,
            # Align to the configured hour so the first run lands after a close,
            # not wherever the container happened to start.
            initial_delay_s=seconds_until_hour(datetime.now(UTC), settings.FETCH_AT_HOUR_UTC),
        )
        scheduler.start()
    elif settings.SCHEDULE_FETCH_ENABLED:
        logger.info("Scheduled pull idle — set FETCH_SYMBOLS to enable it")

    async def _readiness() -> tuple[bool, dict[str, bool]]:
        checks: dict[str, bool] = {}
        try:
            async with engine.connect() as conn:
                # Not just SELECT 1: query the actual table, so a missing
                # schema / unapplied init-db.sql / wrong search_path shows up
                # as "not ready" instead of a 500 on the first fetch.
                await conn.execute(text("SELECT 1 FROM ohlcv LIMIT 1"))
            checks["database"] = True
        except Exception:  # noqa: BLE001
            checks["database"] = False
        checks["redis"] = False
        if redis_client is not None:
            with suppress(Exception):
                await redis_client.ping()
                checks["redis"] = True
        checks["nats"] = nats_client is not None and nats_client.is_connected
        # Database is the hard requirement; redis/nats degrade gracefully.
        return checks["database"], checks

    app.state.readiness_check = _readiness

    yield

    logger.info("Shutting down service", service=settings.SERVICE_NAME)
    if scheduler is not None:
        await scheduler.stop()
    if nats_client is not None:
        with suppress(Exception):
            await nats_client.drain()
    if redis_client is not None:
        with suppress(Exception):
            await redis_client.aclose()
    await engine.dispose()


app = FastAPI(
    title="Market Data Service",
    description="Pobieranie, walidacja i przechowywanie danych OHLCV",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
