"""FastAPI dependencies for backtest."""

from fastapi import HTTPException, Request

from src.core.service import BacktestService


def get_service(request: Request) -> BacktestService:
    """Return the app-wide BacktestService, or 503 if it isn't wired yet."""
    service: BacktestService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="backtest service not ready")
    return service
