from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any

import nats
import redis.asyncio as aredis
import structlog
from fastapi import FastAPI
from sqlalchemy import text
from trading_common.scheduler import PeriodicTask
from trading_common.schemas import Interval

from src.api import router as api_router
from src.config import settings
from src.core.cache import Cache, InMemoryCache, RedisCache
from src.core.catchup import (
    NullSyncMarker,
    RedisSyncMarker,
    coverage_date,
    needs_catchup,
)
from src.core.fetchers import build_default_fetcher
from src.core.incremental import is_weekend
from src.core.observability import setup_observability
from src.core.service import MarketDataService
from src.core.storage import OHLCVRepository
from src.events.publisher import NatsPublisher, NullPublisher, Publisher, ensure_stream
from src.models.db import Base, make_engine, make_sessionmaker

logger = structlog.get_logger()


async def _retire_compression(conn: Any) -> None:
    """Stop TimescaleDB compressing ohlcv, and say so if chunks are still packed.

    TS-1, decided 2026-08-04 by measurement. Compression assumes history is
    immutable and append-only; this table is rewritten by design, because
    adj_close belongs to the bar PLUS every later corporate action, so a split
    makes the provider restate everything and the repair reaches back to
    earliest_timestamp. Writing 20 years of one symbol needed 101429 tuples
    decompressed against a 100000 limit — every history rewrite failed.

    Removing the POLICY is instant and idempotent. DECOMPRESSING existing
    chunks is not: 1043 of them is minutes of work, which would run on every
    container start and blow the health-check budget the way the ml-pipeline
    routes once did. So the remaining chunks are REPORTED with the command that
    clears them, rather than cleared here.
    """
    if not await _has_timescale(conn):
        return
    await conn.execute(
        text("SELECT remove_compression_policy('market_data.ohlcv', if_exists => TRUE)")
    )
    packed = await conn.execute(
        text(
            "SELECT COUNT(*) FROM timescaledb_information.chunks "
            "WHERE hypertable_name = 'ohlcv' AND is_compressed"
        )
    )
    remaining = int(packed.scalar() or 0)
    if remaining:
        logger.warning(
            "ohlcv chunks are still compressed — history rewrites will fail",
            compressed_chunks=remaining,
            fix=("SELECT decompress_chunk(c, true) FROM show_chunks('market_data.ohlcv') c;"),
        )


async def _has_timescale(conn: Any) -> bool:
    """Postgres resolves relation names at PARSE time, so querying a
    timescaledb_information view on a plain Postgres raises rather than
    returning nothing. Gate on the extension itself."""
    found = await conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'"))
    return found.scalar() is not None


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
                # Same trap one level up: `create_all` will not fix the KEY of a
                # table that already exists. A database created before the
                # primary key became (symbol, interval, ts) still carries the
                # old (id, ts), and the bulk upsert needs a unique constraint
                # matching its ON CONFLICT target — without one every write
                # fails with InvalidColumnReferenceError, which reads like a
                # code bug and is a schema age.
                #
                # A hypertable's unique key must include the partition column,
                # which ts is. Idempotent via the catalogue check: ADD
                # CONSTRAINT has no IF NOT EXISTS.
                await conn.execute(
                    text("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint
                            WHERE conrelid = 'ohlcv'::regclass
                              AND conname = 'uq_ohlcv_symbol_interval_ts'
                        ) THEN
                            ALTER TABLE ohlcv
                                ADD CONSTRAINT uq_ohlcv_symbol_interval_ts
                                UNIQUE (symbol, interval, ts);
                        END IF;
                    END $$;
                    """)
                )
                await _retire_compression(conn)
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

        marker: RedisSyncMarker | NullSyncMarker = (
            RedisSyncMarker(redis_client) if redis_client is not None else NullSyncMarker()
        )

        async def _maybe_sync() -> None:
            """One heartbeat: pull only if today's session is not covered yet.

            The question is a DATE comparison, so a suspended host merely
            delays the next beat instead of corrupting the schedule — which an
            elapsed-time timer cannot say for itself.
            """
            now = datetime.now(UTC)
            if not needs_catchup(await marker.last_sync(), now, settings.FETCH_AT_HOUR_UTC):
                return
            logger.info(
                "Sync starting — no completed pull covers the latest session",
                covers=coverage_date(now, settings.FETCH_AT_HOUR_UTC).isoformat(),
            )
            await _sync_job()
            # Recorded only after the run finishes, and against the session it
            # could actually have covered. A crash leaves the day uncovered so
            # the next beat retries; marking first would skip a day of data on
            # the strength of an attempt.
            await marker.mark(coverage_date(datetime.now(UTC), settings.FETCH_AT_HOUR_UTC))

        scheduler = PeriodicTask(
            "market-data-sync",
            interval_s=settings.FETCH_CHECK_INTERVAL_S,
            job=_maybe_sync,
            # Beat immediately: on a stack that runs a few hours a day, waiting
            # even one interval can be the whole window.
            initial_delay_s=0.0,
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
