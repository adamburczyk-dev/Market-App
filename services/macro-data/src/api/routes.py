"""macro-data HTTP API — current macro snapshot / regime + on-demand refresh."""

from datetime import date

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from trading_common.schemas import MacroSnapshot

from src.api.deps import get_service
from src.core.fred_client import DEFAULT_SERIES
from src.core.service import MacroDataService

logger = structlog.get_logger()
router = APIRouter()

# A history request is answered by walking one day at a time, so an unbounded
# range is an unbounded response. 40 years is twice the longest backfill the
# bootstrap script asks for.
MAX_HISTORY_DAYS = 40 * 366


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


# --- history (P2-4) -------------------------------------------------------


class BackfillRequest(BaseModel):
    """Which series to pull, and over what observation window.

    Series are given as FRED ids because that is what the vintage store is keyed
    on; the indicator name is a label for the report.

    The default is DEFAULT_SERIES itself rather than a copy. It was a copy, and
    the copies drifted: switching the fetcher to source series left this list
    still naming FRED's calculated ones, so a backfill went on storing exactly
    the series the change existed to stop using — and reported success.
    """

    series: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_SERIES))
    start: date | None = None
    end: date | None = None


@router.post("/backfill")
async def backfill(
    req: BackfillRequest,
    service: MacroDataService = Depends(get_service),
) -> dict:
    """Pull every VINTAGE of each series from ALFRED and store it.

    Long-running and rate-limited upstream, so it is explicitly requested rather
    than scheduled. Idempotent: the natural key is (series, period, vintage), so
    re-running writes the same rows.
    """
    written = await service.backfill(req.series, req.start, req.end)
    return {"written": written, "total": sum(written.values())}


@router.get("/history")
async def history(
    start: date = Query(...),
    end: date = Query(...),
    service: MacroDataService = Depends(get_service),
) -> dict:
    """Date → regime, each classified from ONLY what was published by that date.

    Days that cannot be classified are absent rather than defaulted: this feeds
    a model feature, and a made-up "expansion" would be a fabricated fact where
    a missing one is the truth.
    """
    if end < start:
        raise HTTPException(status_code=422, detail="end must not precede start")
    if (end - start).days > MAX_HISTORY_DAYS:
        raise HTTPException(
            status_code=422,
            detail=f"range exceeds {MAX_HISTORY_DAYS} days",
        )
    regimes = await service.regime_history(start, end)
    return {"regimes": regimes, "days": len(regimes), "requested": (end - start).days + 1}


@router.get("/observations/{series}")
async def observations(
    series: str,
    start: date | None = None,
    end: date | None = None,
    service: MacroDataService = Depends(get_service),
) -> dict:
    """Raw vintage rows for one series — every revision, not just the latest."""
    rows = await service.series_history(series, start, end)
    return {
        "series": series,
        "count": len(rows),
        "observations": [r.model_dump(mode="json") for r in rows],
    }


@router.get("/coverage")
async def coverage(service: MacroDataService = Depends(get_service)) -> dict:
    """What the panel actually holds, including how much of it is UNDATED.

    Undated rows are invisible to point-in-time reads, so a panel that looks
    full and answers nothing historical would otherwise be indistinguishable
    from one that is genuinely populated.
    """
    return {"series": await service.coverage()}
