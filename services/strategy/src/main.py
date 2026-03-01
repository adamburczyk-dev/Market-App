from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.api import router as api_router
from src.config import settings
from src.core.observability import setup_observability

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting service", service=settings.SERVICE_NAME)
    yield
    logger.info("Shutting down service", service=settings.SERVICE_NAME)


app = FastAPI(
    title="Strategy Service",
    description="Definicja i ewaluacja strategii tradingowych",
    version="0.1.0",
    lifespan=lifespan,
)

setup_observability(app, settings.SERVICE_NAME)
app.include_router(api_router, prefix="/api/v1")
