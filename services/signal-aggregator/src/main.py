from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import nats
import structlog
from fastapi import FastAPI
from trading_common.cost_filter import CostAwareFilter

from src.api import router as api_router
from src.config import settings
from src.core.adaptive_weights import AdaptiveWeightOptimizer
from src.core.company_client import HttpCompanyClient
from src.core.observability import setup_observability
from src.core.service import SignalAggregatorService
from src.events.publisher import NatsPublisher, NullPublisher, Publisher, ensure_stream
from src.events.subscriber import EventSubscriber

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)

    optimizer = AdaptiveWeightOptimizer(
        settings.sources,
        lookback_days=settings.WEIGHT_LOOKBACK_DAYS,
        min_weight=settings.WEIGHT_MIN,
        max_weight=settings.WEIGHT_MAX,
    )

    publisher: Publisher
    nats_client = None
    try:
        nats_client = await nats.connect(settings.NATS_URL)
        js = nats_client.jetstream()
        await ensure_stream(js, settings.NATS_SIGNALS_STREAM, [settings.NATS_SIGNALS_SUBJECTS])
        # Ensure the macro stream too so the regime subscription is start-order independent.
        await ensure_stream(js, settings.NATS_MACRO_STREAM, ["macro.>"])
        publisher = NatsPublisher(js)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS/JetStream unavailable, events disabled", error=str(exc))
        publisher = NullPublisher()
        nats_client = None

    company_client = HttpCompanyClient(settings.COMPANY_CLASSIFIER_URL)
    service = SignalAggregatorService(
        optimizer,
        CostAwareFilter(),
        publisher,
        buy_threshold=settings.BUY_THRESHOLD,
        base_edge_bps=settings.BASE_EDGE_BPS,
        signal_ttl_s=settings.SIGNAL_TTL_SECONDS,
        company_client=company_client,
    )
    app.state.service = service

    subscribers: list[EventSubscriber] = []
    if nats_client is not None:
        try:
            signal_sub = EventSubscriber(
                nats_client.jetstream(),
                settings.NATS_SIGNAL_SUBJECT,
                settings.NATS_SIGNAL_DURABLE,
                service.handle_signal_generated,
                max_deliver=settings.NATS_MAX_DELIVER,
            )
            await signal_sub.start()
            regime_sub = EventSubscriber(
                nats_client.jetstream(),
                settings.NATS_MACRO_SUBJECT,
                settings.NATS_MACRO_DURABLE,
                service.handle_regime_changed,
                max_deliver=settings.NATS_MAX_DELIVER,
            )
            await regime_sub.start()
            subscribers = [signal_sub, regime_sub]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not subscribe to source events", error=str(exc))
            subscribers = []

    async def _readiness() -> tuple[bool, dict[str, bool]]:
        # signal-aggregator consumes + publishes signals → NATS is required.
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
    await company_client.aclose()


app = FastAPI(
    title="Signal Aggregator Service",
    description="Combine ML + rules-based + macro-regime signals into one decision",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
