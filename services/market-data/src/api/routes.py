from datetime import UTC, date, datetime, time

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from trading_common.constants import DEFAULT_SYMBOLS, MAX_OHLCV_LIMIT
from trading_common.schemas import Interval, OHLCVBar

from src.api.deps import get_service
from src.config import settings
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
    limit: int = Query(default=500, ge=1, le=MAX_OHLCV_LIMIT),
    service: MarketDataService = Depends(get_service),
) -> list[OHLCVBar]:
    """Return stored OHLCV bars for a symbol (chronological order).

    The ceiling comes from trading-common, not from a local literal: this route
    is the SERVER side of a limit that callers declare independently, and the
    two drifting apart is invisible until someone asks for the longest window.
    """
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
    except Exception as exc:
        # Storage / cache / event failures used to escape as a bare 500 with an
        # EMPTY body (Starlette answers unhandled errors in plain text), so the
        # caller saw "HTTP 500:" and nothing else. Name the cause instead — this
        # is an internal service, the message is worth more than the opacity.
        logger.exception("Fetch-and-store failed", symbol=symbol, interval=interval.value)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return {"status": "ok", "symbol": symbol.upper(), "interval": interval.value, "rows": rows}


@router.post("/sync")
async def trigger_sync(
    symbols: str | None = Query(default=None, description="CSV; defaults to FETCH_SYMBOLS"),
    interval: Interval = Interval.D1,
    service: MarketDataService = Depends(get_service),
) -> dict:
    """Run the incremental pull now — the same job the scheduler runs.

    Each symbol resumes from its newest stored bar, so this is also the repair
    button after downtime: the gap is whatever it is, and one call closes it.
    """
    wanted = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols
        else settings.fetch_symbols
    )
    if not wanted:
        raise HTTPException(
            status_code=400,
            detail="no symbols: pass ?symbols=AAPL,MSFT or configure FETCH_SYMBOLS",
        )
    try:
        return await service.sync_universe(
            wanted,
            interval,
            pause_s=settings.FETCH_SYMBOL_PAUSE_S,
            overlap_days=settings.FETCH_OVERLAP_DAYS,
            initial_history_days=settings.FETCH_INITIAL_HISTORY_DAYS,
        )
    except Exception as exc:
        logger.exception("Sync failed", symbols=len(wanted))
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.get("/sync/status")
async def sync_status(
    symbols: str | None = Query(default=None, description="CSV; defaults to FETCH_SYMBOLS"),
    interval: Interval = Interval.D1,
    service: MarketDataService = Depends(get_service),
) -> dict:
    """How stale each symbol is — the answer to "did the schedule actually run".

    A scheduler that silently stopped looks exactly like one that ran and found
    nothing, until a feature window comes up short weeks later.
    """
    wanted = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols
        else settings.fetch_symbols
    )
    now = datetime.now(UTC)
    entries = []
    for symbol in wanted:
        latest = await service.latest_timestamp(symbol, interval)
        entries.append(
            {
                "symbol": symbol,
                "latest": latest.isoformat() if latest else None,
                "age_days": round((now - latest).total_seconds() / 86_400, 2) if latest else None,
            }
        )
    stale = [e for e in entries if e["latest"] is None or float(e["age_days"] or 0) > 5]
    return {
        "as_of": now.isoformat(),
        "scheduled": settings.SCHEDULE_FETCH_ENABLED and bool(settings.fetch_symbols),
        "symbols": entries,
        "stale": [e["symbol"] for e in stale],
    }


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
