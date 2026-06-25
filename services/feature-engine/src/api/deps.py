"""FastAPI dependencies for feature-engine."""

from fastapi import HTTPException, Request

from src.core.service import FeatureEngineService


def get_service(request: Request) -> FeatureEngineService:
    """Return the app-wide FeatureEngineService, or 503 if it isn't wired yet."""
    service: FeatureEngineService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="feature-engine service not ready")
    return service
