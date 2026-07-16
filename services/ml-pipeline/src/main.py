from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import nats
import structlog
from fastapi import FastAPI

from src.api import router as api_router
from src.config import settings
from src.core.feature_client import HttpFeatureClient
from src.core.macro_client import HttpMacroClient
from src.core.market_data_client import HttpMarketDataClient
from src.core.model_store import MlflowModelStore
from src.core.monitoring.drift_detector import DriftDetector
from src.core.observability import setup_observability
from src.core.registry import ModelRegistry
from src.core.service import MLPipelineService
from src.core.serving import ServingEngine
from src.events.publisher import NatsPublisher, NullPublisher, Publisher, ensure_stream
from src.events.subscriber import EventSubscriber

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)

    detector = DriftDetector()
    registry = ModelRegistry()
    market_client = HttpMarketDataClient(settings.MARKET_DATA_URL)

    model_store: MlflowModelStore | None
    try:
        # sqlite backend needs its parent directory to exist
        db_path = settings.MLFLOW_TRACKING_URI.removeprefix("sqlite:///")
        if db_path != settings.MLFLOW_TRACKING_URI:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        model_store = MlflowModelStore(settings.MLFLOW_TRACKING_URI, model_name=settings.MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow store unavailable — training runs won't persist", error=str(exc))
        model_store = None

    publisher: Publisher
    nats_client = None
    try:
        nats_client = await nats.connect(settings.NATS_URL)
        js = nats_client.jetstream()
        await ensure_stream(js, settings.NATS_ML_STREAM, [settings.NATS_ML_SUBJECTS])
        # source stream for the inference trigger (start-order independent)
        await ensure_stream(js, settings.NATS_FEATURES_STREAM, ["features.>"])
        publisher = NatsPublisher(js)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS/JetStream unavailable, events disabled", error=str(exc))
        publisher = NullPublisher()
        nats_client = None

    feature_client = HttpFeatureClient(settings.FEATURE_ENGINE_URL)
    macro_client = HttpMacroClient(settings.MACRO_DATA_URL)
    serving = ServingEngine(
        publisher,
        feature_client,
        macro_client,
        model_store,
        buy_threshold=settings.BUY_PROBABILITY,
        sell_threshold=settings.SELL_PROBABILITY,
        serve_interval=settings.SERVE_INTERVAL,
        horizon_days=settings.LABEL_HORIZON_DAYS,
    )
    if serving.reload() is None:
        logger.info("Serving inactive until a model version is promoted")

    service = MLPipelineService(
        detector,
        registry,
        publisher,
        market_client=market_client,
        model_store=model_store,
        serving=serving,
    )
    app.state.service = service

    subscriber: EventSubscriber | None = None
    if nats_client is not None:
        try:
            subscriber = EventSubscriber(
                nats_client.jetstream(),
                settings.NATS_FEATURES_SUBJECT,
                settings.NATS_FEATURES_DURABLE,
                serving.handle_features_ready,
                max_deliver=settings.NATS_MAX_DELIVER,
            )
            await subscriber.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not subscribe to features events", error=str(exc))
            subscriber = None

    async def _readiness() -> tuple[bool, dict[str, bool]]:
        # ml-pipeline publishes drift events → NATS is required.
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
    await market_client.aclose()
    await feature_client.aclose()
    await macro_client.aclose()


app = FastAPI(
    title="ML Pipeline Service",
    description="Training, inference, model registry i drift detection",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
