"""Observability — Prometheus metrics + structured logging dla wszystkich serwisów."""

import structlog
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def setup_observability(app: FastAPI, service_name: str) -> None:
    """Konfiguruj observability — wywołaj raz w main.py."""

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if app.debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
    )

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    @app.get("/health", tags=["ops"])
    async def health() -> dict:
        return {"status": "healthy", "service": service_name}

    @app.get("/ready", tags=["ops"])
    async def readiness() -> dict:
        # TODO: sprawdź zależności (DB, Redis, NATS)
        return {"status": "ready", "service": service_name}
