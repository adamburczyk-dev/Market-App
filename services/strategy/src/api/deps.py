"""FastAPI dependencies for strategy."""

from fastapi import HTTPException, Request

from src.core.service import StrategyService


def get_service(request: Request) -> StrategyService:
    """Return the app-wide StrategyService, or 503 if it isn't wired yet."""
    service: StrategyService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="strategy service not ready")
    return service
