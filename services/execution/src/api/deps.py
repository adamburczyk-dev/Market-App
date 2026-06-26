"""FastAPI dependencies for execution."""

from fastapi import HTTPException, Request

from src.core.service import ExecutionService


def get_service(request: Request) -> ExecutionService:
    """Return the app-wide ExecutionService, or 503 if it isn't wired yet."""
    service: ExecutionService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="execution service not ready")
    return service
