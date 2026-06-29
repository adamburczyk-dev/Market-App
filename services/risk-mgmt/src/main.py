from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import nats
import redis.asyncio as aredis
import structlog
from fastapi import FastAPI

from src.api import router as api_router
from src.config import settings
from src.core.circuit_breaker import CircuitBreaker
from src.core.observability import setup_observability
from src.core.portfolio import PortfolioState
from src.core.repository import NullStateRepository, RedisStateRepository, StateRepository
from src.core.service import RiskMgmtService
from src.core.sizing import PositionSizer
from src.events.publisher import NatsPublisher, NullPublisher, Publisher, ensure_stream
from src.events.subscriber import SignalSubscriber

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)

    sizer = PositionSizer(
        base_risk_per_trade=settings.BASE_RISK_PER_TRADE,
        dd_scaling_start=settings.DD_SCALING_START,
        dd_scaling_end=settings.DD_SCALING_END,
        max_position_pct=settings.MAX_POSITION_PCT,
    )
    breaker = CircuitBreaker(
        drawdown_warn_pct=settings.DRAWDOWN_WARN_PCT,
        daily_loss_halt_pct=settings.DAILY_LOSS_HALT_PCT,
        drawdown_flatten_pct=settings.DRAWDOWN_FLATTEN_PCT,
    )
    portfolio = PortfolioState(value=settings.PORTFOLIO_VALUE)

    repository: StateRepository
    redis_client = None
    try:
        redis_client = aredis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        repository = RedisStateRepository(redis_client)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis unavailable, portfolio state not persisted", error=str(exc))
        repository = NullStateRepository()
        redis_client = None

    publisher: Publisher
    nats_client = None
    subscriber: SignalSubscriber | None = None
    try:
        nats_client = await nats.connect(settings.NATS_URL)
        js = nats_client.jetstream()
        await ensure_stream(js, settings.NATS_ORDERS_STREAM, [settings.NATS_ORDERS_SUBJECTS])
        await ensure_stream(js, settings.NATS_RISK_STREAM, [settings.NATS_RISK_SUBJECTS])
        await ensure_stream(js, settings.NATS_SOURCE_STREAM, ["signal.>"])
        publisher = NatsPublisher(js)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS/JetStream unavailable, events disabled", error=str(exc))
        publisher = NullPublisher()
        nats_client = None

    service = RiskMgmtService(publisher, sizer, breaker, portfolio, repository)
    await service.restore()
    app.state.service = service

    if nats_client is not None:
        try:
            subscriber = SignalSubscriber(
                nats_client.jetstream(),
                settings.NATS_SOURCE_SUBJECT,
                settings.NATS_DURABLE,
                service.handle_signal_event,
                max_deliver=settings.NATS_MAX_DELIVER,
            )
            await subscriber.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not subscribe to signal events", error=str(exc))
            subscriber = None

    async def _readiness() -> tuple[bool, dict[str, bool]]:
        # risk-mgmt reacts to signal events → NATS is required.
        nats_ok = nats_client is not None and nats_client.is_connected
        return nats_ok, {"nats": nats_ok}

    app.state.readiness_check = _readiness

    yield

    logger.info("Shutting down service", service=settings.SERVICE_NAME)
    if subscriber is not None:
        await subscriber.stop()
    if nats_client is not None:
        with suppress(Exception):
            await nats_client.drain()
    if redis_client is not None:
        with suppress(Exception):
            await redis_client.aclose()


app = FastAPI(
    title="Risk Management Service",
    description="Position sizing, portfolio optimization i risk metrics",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
