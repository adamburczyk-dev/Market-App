from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import nats
import redis.asyncio as aredis
import structlog
from fastapi import FastAPI

from src.api import router as api_router
from src.config import settings
from src.core.cache import Cache, InMemoryCache, RedisCache
from src.core.fetchers import build_default_fetcher
from src.core.observability import setup_observability
from src.core.service import MarketDataService
from src.core.storage import OHLCVRepository
from src.events.publisher import NatsPublisher, NullPublisher, Publisher
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
        publisher = NatsPublisher(nats_client)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS unavailable, event publishing disabled", error=str(exc))
        publisher = NullPublisher()
        nats_client = None

    fetcher = build_default_fetcher(settings)
    app.state.service = MarketDataService(fetcher, repository, cache, publisher)

    yield

    logger.info("Shutting down service", service=settings.SERVICE_NAME)
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
