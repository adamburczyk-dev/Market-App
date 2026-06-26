"""FastAPI dependencies for risk-mgmt."""

from fastapi import HTTPException, Request

from src.core.service import RiskMgmtService


def get_service(request: Request) -> RiskMgmtService:
    """Return the app-wide RiskMgmtService, or 503 if it isn't wired yet."""
    service: RiskMgmtService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="risk-mgmt service not ready")
    return service
