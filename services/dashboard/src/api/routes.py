"""dashboard HTTP API — one endpoint per section + a self-contained HTML page.

Sections are separate endpoints rather than one fat payload on purpose: the
health probe and the correlation grid cost real upstream work, and a page that
refreshes its portfolio numbers every few seconds must not drag those along.
"""

from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse

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


@router.get("/sections/portfolio")
async def portfolio_section(service: DashboardService = Depends(get_service)) -> dict[str, Any]:
    """Section 1 — equity curve, P&L, open positions."""
    return await service.portfolio_section()


@router.get("/sections/risk")
async def risk_section(service: DashboardService = Depends(get_service)) -> dict[str, Any]:
    """Section 2 — VaR, drawdown path, correlation grid of what is held."""
    return await service.risk_section()


@router.get("/sections/strategy")
async def strategy_section(service: DashboardService = Depends(get_service)) -> dict[str, Any]:
    """Section 3 — per-strategy status and learned decision weight."""
    return await service.strategy_section()


@router.post("/sections/backtest")
async def backtest_section(
    strategy: str = Query(...),
    symbol: str = Query(...),
    limit: int = Query(default=500, ge=10, le=10_000),
    service: DashboardService = Depends(get_service),
) -> JSONResponse:
    """Section 4 — run a backtest on demand and return its curve.

    A POST because it does work. The upstream status is passed through: 404
    means the strategy name is unknown, 422 that it needs a universe backtest,
    and flattening either into 200-with-nothing would misreport a working
    service as a broken one.
    """
    status_code, body = await service.backtest_section(strategy, symbol.upper(), limit)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/sections/ml")
async def ml_section(service: DashboardService = Depends(get_service)) -> dict[str, Any]:
    """Section 5 — registered models, last long-running runs, serving state."""
    return await service.ml_section()


@router.get("/sections/health")
async def health_section(service: DashboardService = Depends(get_service)) -> dict[str, Any]:
    """Section 6 — per-service up/down and measured latency."""
    return await service.health_section()


@router.get("/ui", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    """Self-contained HTML dashboard; its JS polls the sibling section routes."""
    return HTMLResponse(INDEX_HTML)
