"""FastAPI dependencies for market-data."""

from fastapi import HTTPException, Request

from src.core.service import MarketDataService


def get_service(request: Request) -> MarketDataService:
    """Return the app-wide MarketDataService, or 503 if it isn't wired yet."""
    service: MarketDataService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="market-data service not ready")
    return service
