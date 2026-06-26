from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import nats
import structlog
from fastapi import FastAPI

from src.api import router as api_router
from src.config import settings
from src.core.observability import setup_observability
from src.core.paper_broker import PaperBroker
from src.core.risk_client import HttpRiskClient
from src.core.service import ExecutionService
from src.events.publisher import NatsPublisher, NullPublisher, Publisher, ensure_stream
from src.events.subscriber import OrderSubscriber

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)

    broker = PaperBroker(initial_cash=settings.INITIAL_CASH, slippage_bps=settings.SLIPPAGE_BPS)
    risk_client = HttpRiskClient(settings.RISK_MGMT_URL)

    publisher: Publisher
    nats_client = None
    subscriber: OrderSubscriber | None = None
    try:
        nats_client = await nats.connect(settings.NATS_URL)
        js = nats_client.jetstream()
        await ensure_stream(js, settings.NATS_ORDERS_STREAM, [settings.NATS_ORDERS_SUBJECTS])
        publisher = NatsPublisher(js)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS/JetStream unavailable, events disabled", error=str(exc))
        publisher = NullPublisher()
        nats_client = None

    service = ExecutionService(broker, publisher, risk_client)
    app.state.service = service

    if nats_client is not None:
        try:
            subscriber = OrderSubscriber(
                nats_client.jetstream(),
                settings.NATS_SOURCE_SUBJECT,
                settings.NATS_DURABLE,
                service.handle_order_event,
                max_deliver=settings.NATS_MAX_DELIVER,
            )
            await subscriber.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not subscribe to order events", error=str(exc))
            subscriber = None

    async def _readiness() -> tuple[bool, dict[str, bool]]:
        # execution reacts to order events → NATS is required.
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
    await risk_client.aclose()


app = FastAPI(
    title="Execution Service",
    description="Paper i live trading, order management",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
