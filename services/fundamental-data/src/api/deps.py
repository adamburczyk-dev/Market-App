"""FastAPI dependencies for fundamental-data."""

from fastapi import HTTPException, Request

from src.core.service import FundamentalDataService


def get_service(request: Request) -> FundamentalDataService:
    """Return the app-wide FundamentalDataService, or 503 if it isn't wired yet."""
    service: FundamentalDataService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="fundamental-data service not ready")
    return service
