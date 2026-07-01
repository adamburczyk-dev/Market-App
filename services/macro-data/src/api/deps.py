"""FastAPI dependencies for macro-data."""

from fastapi import HTTPException, Request

from src.core.service import MacroDataService


def get_service(request: Request) -> MacroDataService:
    """Return the app-wide MacroDataService, or 503 if it isn't wired yet."""
    service: MacroDataService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="macro-data service not ready")
    return service
