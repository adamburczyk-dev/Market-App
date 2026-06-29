"""FastAPI dependencies for notification."""

from fastapi import HTTPException, Request

from src.core.service import NotificationService


def get_service(request: Request) -> NotificationService:
    """Return the app-wide NotificationService, or 503 if it isn't wired yet."""
    service: NotificationService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="notification service not ready")
    return service
