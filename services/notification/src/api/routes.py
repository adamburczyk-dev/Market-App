"""notification HTTP API — channel introspection, recent alerts, manual test alert."""

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import get_service
from src.core.channels import Alert
from src.core.service import NotificationService

logger = structlog.get_logger()
router = APIRouter()


class TestAlertRequest(BaseModel):
    severity: str = "info"
    title: str = "Test alert"
    message: str = "Manual test from /test-alert"


def _alert_dict(a: Alert) -> dict:
    return {
        "severity": a.severity,
        "title": a.title,
        "message": a.message,
        "source": a.source,
        "metadata": a.metadata,
    }


@router.get("/status")
async def status() -> dict:
    return {"service": "notification", "status": "ready"}


@router.get("/channels")
async def channels(service: NotificationService = Depends(get_service)) -> dict:
    return {"channels": [ch.name for ch in service.channels]}


@router.get("/alerts/recent")
async def recent_alerts(
    limit: int = 20, service: NotificationService = Depends(get_service)
) -> dict:
    return {"alerts": [_alert_dict(a) for a in service.recent(limit)]}


@router.post("/test-alert")
async def test_alert(
    req: TestAlertRequest, service: NotificationService = Depends(get_service)
) -> dict:
    """Dispatch a synthetic alert through all channels (ops smoke test)."""
    alert = Alert(severity=req.severity, title=req.title, message=req.message, source="manual.test")
    await service.dispatch(alert)
    return {"dispatched": True, "channels": [ch.name for ch in service.channels]}
