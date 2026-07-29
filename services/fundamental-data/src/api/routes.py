"""fundamental-data HTTP API — fundamentals + Piotroski F-score per symbol."""

from datetime import date

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
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


@router.get("/fundamentals/{symbol}/as-of")
async def get_as_of(
    symbol: str,
    session: date = Query(description="Session date; only filings published BEFORE it count"),
    service: FundamentalDataService = Depends(get_service),
) -> dict:
    """Point-in-time read — what was knowable about `symbol` when `session` opened.

    Deliberately excludes a filing dated the session itself: filings land after
    the close, so counting one as known during that session is intraday
    look-ahead. A statement with no filing date is never returned — undated is
    not old.
    """
    statement = await service.as_of_session(symbol, session)
    if statement is None:
        raise HTTPException(
            status_code=404,
            detail=f"nothing was published for {symbol.upper()} before {session.isoformat()}",
        )
    return {"statement": statement.model_dump(mode="json"), "as_of": session.isoformat()}


@router.get("/panel")
async def get_panel(
    symbols: str = Query(description="csv of tickers"),
    service: FundamentalDataService = Depends(get_service),
) -> dict:
    """Every stored period for these symbols — training's single fetch.

    The caller does the as-of join locally (one request instead of
    symbols x sessions round trips) using the SAME rule this service applies:
    `trading_common.fundamentals.latest_available_before`.
    """
    requested = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    statements = await service.panel(requested)
    dated = sum(1 for s in statements if s.filed_at is not None)
    return {
        "symbols": requested,
        "rows": len(statements),
        # An undated row is invisible to every point-in-time read, so the count
        # is reported rather than left for the caller to discover as missing data.
        "rows_without_filed_at": len(statements) - dated,
        "statements": [s.model_dump(mode="json") for s in statements],
    }


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
