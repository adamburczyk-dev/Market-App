"""fundamental-data HTTP API — fundamentals + Piotroski F-score per symbol."""

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from trading_common.schemas import FinancialStatements

from src.api.deps import get_service
from src.core.piotroski import FScoreBreakdown
from src.core.service import FundamentalDataService

logger = structlog.get_logger()
router = APIRouter()


class IngestRequest(BaseModel):
    """Manually-provided statements (works without SEC access)."""

    current: FinancialStatements
    prior: FinancialStatements | None = None


def _view(statement: FinancialStatements, breakdown: FScoreBreakdown) -> dict:
    return {
        "statement": statement.model_dump(mode="json"),
        "f_score": breakdown.score,
        "f_score_breakdown": breakdown.as_dict(),
    }


@router.get("/status")
async def status() -> dict:
    return {"service": "fundamental-data", "status": "ready"}


@router.get("/fundamentals")
async def list_symbols(service: FundamentalDataService = Depends(get_service)) -> dict:
    return {"symbols": service.symbols()}


@router.get("/fundamentals/{symbol}")
async def get_fundamentals(
    symbol: str, service: FundamentalDataService = Depends(get_service)
) -> dict:
    record = service.get(symbol)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no fundamentals for {symbol}")
    return _view(*record)


@router.post("/refresh/{symbol}")
async def refresh(symbol: str, service: FundamentalDataService = Depends(get_service)) -> dict:
    """Pull the latest annual filings from EDGAR, score, and publish."""
    record = await service.refresh(symbol)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"no EDGAR fundamentals for {symbol} (SEC_USER_AGENT set? ticker known?)",
        )
    return _view(*record)


@router.post("/statements")
async def ingest(
    req: IngestRequest, service: FundamentalDataService = Depends(get_service)
) -> dict:
    """Score and publish manually-provided statements."""
    record = await service.ingest(req.current, req.prior)
    return _view(*record)
