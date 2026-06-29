"""FastAPI dependencies for ml-pipeline."""

from fastapi import HTTPException, Request

from src.core.service import MLPipelineService


def get_service(request: Request) -> MLPipelineService:
    """Return the app-wide MLPipelineService, or 503 if it isn't wired yet."""
    service: MLPipelineService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="ml-pipeline service not ready")
    return service
