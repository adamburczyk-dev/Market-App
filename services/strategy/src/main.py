from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import nats
import structlog
from fastapi import FastAPI
from trading_common.cost_filter import CostAwareFilter
from trading_common.risk_envelope import RiskEnvelope

# Importing the package registers every built-in rule — the service does not
# keep its own list of what exists.
from trading_common.strategies import StrategyRule, all_strategies, get_strategy

from src.api import router as api_router
from src.config import settings
from src.core.feature_client import HttpFeatureClient
from src.core.observability import setup_observability
from src.core.portfolio_client import HttpPortfolioClient
from src.core.service import PortfolioSnapshot, StrategyService
from src.events.publisher import NatsPublisher, NullPublisher, Publisher, ensure_stream
from src.events.subscriber import EventSubscriber

logger = structlog.get_logger()


def selected_rules() -> list[StrategyRule]:
    """Rules named by ENABLED_STRATEGIES, or every registered rule when empty.

    An unknown name raises: a misspelled strategy that merely does not run is
    indistinguishable from one that ran and found no setup, and this service's
    whole job is to be the thing that fires.
    """
    names = settings.enabled_strategies
    if not names:
        return all_strategies()
    return [get_strategy(name) for name in names]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)

    client = HttpFeatureClient(settings.FEATURE_ENGINE_URL)
    portfolio_client = HttpPortfolioClient(settings.RISK_MGMT_URL)
    rules = selected_rules()
    logger.info("Strategies enabled", strategies=[r.name for r in rules])

    publisher: Publisher
    nats_client = None
    subscribers: list[EventSubscriber] = []
    try:
        nats_client = await nats.connect(settings.NATS_URL)
        js = nats_client.jetstream()
        await ensure_stream(js, settings.NATS_SIGNALS_STREAM, [settings.NATS_SIGNALS_SUBJECTS])
        await ensure_stream(js, settings.NATS_STRATEGY_STREAM, [settings.NATS_STRATEGY_SUBJECTS])
        await ensure_stream(js, settings.NATS_SOURCE_STREAM, ["features.>"])
        await ensure_stream(js, settings.NATS_BACKTEST_STREAM, ["backtest.>"])
        publisher = NatsPublisher(js)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NATS/JetStream unavailable, events disabled", error=str(exc))
        publisher = NullPublisher()
        nats_client = None

    service = StrategyService(
        client,
        publisher,
        rules,
        RiskEnvelope(),
        CostAwareFilter(),
        PortfolioSnapshot(
            value=settings.PORTFOLIO_VALUE,
            exposure_pct=settings.CURRENT_EXPOSURE_PCT,
            drawdown_pct=settings.CURRENT_DRAWDOWN_PCT,
            daily_loss_pct=settings.DAILY_LOSS_PCT,
        ),
        rule_params=settings.STRATEGY_PARAMS,
        fallback_stop_pct=settings.STOP_LOSS_PCT,
        expected_edge_bps=settings.EXPECTED_EDGE_BPS,
        market_cap_tier=settings.MARKET_CAP_TIER,
        portfolio_client=portfolio_client,
    )
    app.state.service = service

    if nats_client is not None:
        try:
            features_sub = EventSubscriber(
                nats_client.jetstream(),
                settings.NATS_SOURCE_SUBJECT,
                settings.NATS_DURABLE,
                service.handle_features_ready_event,
                max_deliver=settings.NATS_MAX_DELIVER,
            )
            await features_sub.start()
            revalidation_sub = EventSubscriber(
                nats_client.jetstream(),
                settings.NATS_BACKTEST_SUBJECT,
                settings.NATS_BACKTEST_DURABLE,
                service.handle_revalidated_event,
                max_deliver=settings.NATS_MAX_DELIVER,
            )
            await revalidation_sub.start()
            subscribers = [features_sub, revalidation_sub]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not subscribe to source events", error=str(exc))
            subscribers = []

    async def _readiness() -> tuple[bool, dict[str, bool]]:
        # strategy's job is reacting to features events → NATS is required.
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
    await client.aclose()
    await portfolio_client.aclose()


app = FastAPI(
    title="Strategy Service",
    description="Definicja i ewaluacja strategii tradingowych",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
