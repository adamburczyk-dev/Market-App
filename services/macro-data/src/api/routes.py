"""macro-data HTTP API — current macro snapshot / regime + on-demand refresh."""

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from trading_common.schemas import MacroSnapshot

from src.api.deps import get_service
from src.core.service import MacroDataService

logger = structlog.get_logger()
router = APIRouter()


class RefreshRequest(BaseModel):
    """Optional manual indicators; merged over FRED-fetched values (overrides win).

    Lets the service run without a FRED key and supply indicators FRED doesn't
    serve here (PMI, CPI YoY).
    """

    yield_curve_10y_2y: float | None = None
    credit_spread_baa_10y: float | None = None
    pmi: float | None = None
    cpi_yoy: float | None = None
    unemployment_rate: float | None = None
    fed_funds_rate: float | None = Field(default=None)


@router.get("/status")
async def status() -> dict:
    return {"service": "macro-data", "status": "ready"}


@router.get("/snapshot", response_model=MacroSnapshot)
async def snapshot(service: MacroDataService = Depends(get_service)) -> MacroSnapshot:
    """Latest macro snapshot; 404 until the first refresh has run."""
    snap = service.snapshot
    if snap is None:
        raise HTTPException(status_code=404, detail="no macro snapshot yet — call POST /refresh")
    return snap


@router.get("/regime")
async def regime(service: MacroDataService = Depends(get_service)) -> dict:
    r = service.regime
    return {"regime": r.value if r is not None else None}


@router.post("/refresh", response_model=MacroSnapshot)
async def refresh(
    req: RefreshRequest,
    service: MacroDataService = Depends(get_service),
) -> MacroSnapshot:
    """Fetch indicators (FRED + overrides), reclassify the regime, publish events."""
    return await service.refresh(overrides=req.model_dump())
