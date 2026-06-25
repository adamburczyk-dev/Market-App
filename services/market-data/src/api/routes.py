from datetime import UTC, date, datetime, time

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from trading_common.constants import DEFAULT_SYMBOLS
from trading_common.schemas import Interval, OHLCVBar

from src.api.deps import get_service
from src.core.fetchers.base import FetchError
from src.core.service import MarketDataService

logger = structlog.get_logger()
router = APIRouter()


def _start_of_day(d: date) -> datetime:
    return datetime.combine(d, time.min, tzinfo=UTC)


def _end_of_day(d: date) -> datetime:
    return datetime.combine(d, time.max, tzinfo=UTC)


@router.get("/ohlcv/{symbol}", response_model=list[OHLCVBar])
async def get_ohlcv(
    symbol: str,
    interval: Interval = Interval.D1,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
    service: MarketDataService = Depends(get_service),
) -> list[OHLCVBar]:
    """Return stored OHLCV bars for a symbol (chronological order)."""
    start = _start_of_day(start_date) if start_date else None
    end = _end_of_day(end_date) if end_date else None
    return await service.get_ohlcv(symbol.upper(), interval, start, end, limit)


@router.post("/fetch/{symbol}")
async def trigger_fetch(
    symbol: str,
    interval: Interval = Interval.D1,
    start_date: date | None = None,
    end_date: date | None = None,
    service: MarketDataService = Depends(get_service),
) -> dict:
    """Fetch fresh data from the source, store it, and publish MarketDataUpdatedEvent."""
    start = _start_of_day(start_date) if start_date else None
    end = _end_of_day(end_date) if end_date else None
    try:
        rows = await service.fetch_and_store(symbol.upper(), interval, start, end)
    except FetchError as exc:
        logger.error("Fetch failed", symbol=symbol, error=str(exc))
        raise HTTPException(status_code=502, detail=f"fetch failed: {exc}") from exc
    return {"status": "ok", "symbol": symbol.upper(), "interval": interval.value, "rows": rows}


@router.get("/symbols")
async def list_symbols(request: Request) -> dict:
    """List symbols present in storage; fall back to defaults when empty/unavailable."""
    service: MarketDataService | None = getattr(request.app.state, "service", None)
    if service is not None:
        try:
            symbols = await service.list_symbols()
            if symbols:
                return {"symbols": symbols}
        except Exception as exc:  # noqa: BLE001 - fall back to defaults on any storage error
            logger.warning("list_symbols failed, using defaults", error=str(exc))
    return {"symbols": list(DEFAULT_SYMBOLS)}
