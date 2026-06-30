"""FastAPI dependencies for dashboard."""

from fastapi import HTTPException, Request

from src.core.service import DashboardService


def get_service(request: Request) -> DashboardService:
    """Return the app-wide DashboardService, or 503 if it isn't wired yet."""
    service: DashboardService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="dashboard service not ready")
    return service
