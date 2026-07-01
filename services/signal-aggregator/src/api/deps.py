"""FastAPI dependencies for signal-aggregator."""

from fastapi import HTTPException, Request

from src.core.service import SignalAggregatorService


def get_service(request: Request) -> SignalAggregatorService:
    """Return the app-wide SignalAggregatorService, or 503 if it isn't wired yet."""
    service: SignalAggregatorService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="signal-aggregator service not ready")
    return service
