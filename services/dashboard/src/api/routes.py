"""dashboard HTTP API — aggregated overview JSON + a self-contained HTML page."""

from typing import Any

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from src.api.deps import get_service
from src.api.ui import INDEX_HTML
from src.core.service import DashboardService

logger = structlog.get_logger()
router = APIRouter()


@router.get("/status")
async def status() -> dict:
    return {"service": "dashboard", "status": "ready"}


@router.get("/overview")
async def overview(service: DashboardService = Depends(get_service)) -> dict[str, Any]:
    """Aggregated, partial-tolerant view of the whole system."""
    return await service.overview()


@router.get("/ui", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    """Self-contained HTML dashboard; its JS polls ./overview."""
    return HTMLResponse(INDEX_HTML)
