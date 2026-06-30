from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

import structlog
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.api import router as api_router
from src.config import settings
from src.core.clients import HttpDashboardSource
from src.core.observability import setup_observability
from src.core.service import DashboardService

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)

    source = HttpDashboardSource(
        risk_url=settings.RISK_MGMT_URL,
        execution_url=settings.EXECUTION_URL,
        notification_url=settings.NOTIFICATION_URL,
        ml_url=settings.ML_PIPELINE_URL,
    )
    service = DashboardService(source)
    app.state.service = service

    async def _readiness() -> tuple[bool, dict[str, bool]]:
        # The dashboard is a read-only aggregator: it is ready as soon as it is up
        # (it tolerates missing upstreams), but reports per-source reachability.
        overview = await service.overview()
        checks = {name: status == "ok" for name, status in overview["sources"].items()}
        return True, checks

    app.state.readiness_check = _readiness

    yield

    logger.info("Shutting down service", service=settings.SERVICE_NAME)
    with suppress(Exception):
        await source.aclose()


app = FastAPI(
    title="Dashboard Service",
    description="UI dashboard dla systemu tradingowego",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def index() -> RedirectResponse:
    return RedirectResponse(url="/api/v1/dashboard/ui")


setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
