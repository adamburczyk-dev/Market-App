from datetime import date

import structlog
from fastapi import APIRouter, HTTPException, Query
from trading_common.schemas import Interval, OHLCVBar

logger = structlog.get_logger()
router = APIRouter()


@router.get("/ohlcv/{symbol}", response_model=list[OHLCVBar])
async def get_ohlcv(
    symbol: str,
    interval: Interval = Interval.D1,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = Query(default=100, le=5000),
) -> list[OHLCVBar]:
    """Pobierz dane OHLCV dla symbolu."""
    logger.info("Fetching OHLCV", symbol=symbol, interval=interval)
    # TODO: implementacja w tygodniu 2
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/fetch/{symbol}", status_code=202)
async def trigger_fetch(symbol: str, interval: Interval = Interval.D1) -> dict:
    """Wyzwól asynchroniczne pobranie danych (publish event po zakończeniu)."""
    logger.info("Triggering fetch", symbol=symbol, interval=interval)
    # TODO: implementacja w tygodniu 2
    return {"status": "accepted", "symbol": symbol, "interval": interval}


@router.get("/symbols")
async def list_symbols() -> dict:
    """Lista dostępnych symboli."""
    return {"symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]}
