import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from trading_common.schemas import FeatureVector, Interval

from src.api.deps import get_service
from src.core.service import FeatureEngineService

logger = structlog.get_logger()
router = APIRouter()


@router.post("/compute/{symbol}", response_model=FeatureVector)
async def compute(
    symbol: str,
    interval: Interval = Interval.D1,
    service: FeatureEngineService = Depends(get_service),
) -> FeatureVector:
    """Fetch OHLCV from market-data, compute features, store + publish event."""
    try:
        fv = await service.compute_for_symbol(symbol.upper(), interval)
    except httpx.HTTPError as exc:
        logger.error("market-data query failed", symbol=symbol, error=str(exc))
        raise HTTPException(status_code=502, detail=f"market-data query failed: {exc}") from exc
    if fv is None:
        raise HTTPException(status_code=404, detail="no market data / not enough bars")
    return fv


@router.get("/features/{symbol}", response_model=FeatureVector)
async def get_features(
    symbol: str,
    interval: Interval = Interval.D1,
    service: FeatureEngineService = Depends(get_service),
) -> FeatureVector:
    """Return the latest computed features for a symbol."""
    fv = service.get_features(symbol.upper(), interval)
    if fv is None:
        raise HTTPException(status_code=404, detail="no features computed yet")
    return fv


@router.get("/features")
async def list_features(request: Request) -> dict:
    """List symbols that currently have computed features."""
    service: FeatureEngineService | None = getattr(request.app.state, "service", None)
    if service is None:
        return {"symbols": []}
    return {"symbols": service.list_symbols()}
