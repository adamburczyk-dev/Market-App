"""Observability — Prometheus metrics + structured logging dla wszystkich serwisów."""

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse
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
    async def readiness() -> JSONResponse:
        # main.py may register app.state.readiness_check: () -> (ready, checks).
        check = getattr(app.state, "readiness_check", None)
        if check is None:
            return JSONResponse({"status": "ready", "service": service_name})
        ready, checks = await check()
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "not_ready",
                "service": service_name,
                "checks": checks,
            },
        )
