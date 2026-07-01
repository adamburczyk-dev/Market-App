"""FastAPI dependencies for company-classifier."""

from fastapi import HTTPException, Request

from src.core.service import CompanyClassifierService


def get_service(request: Request) -> CompanyClassifierService:
    """Return the app-wide CompanyClassifierService, or 503 if it isn't wired yet."""
    service: CompanyClassifierService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="company-classifier service not ready")
    return service
