import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from trading_common.schemas import Interval

from src.api.deps import get_service
from src.core.service import StrategyService

logger = structlog.get_logger()
router = APIRouter()


class DecayMetrics(BaseModel):
    sharpe_30d: float
    sharpe_90d: float
    sharpe_180d: float
    win_rate_30d: float
    profit_factor_30d: float
    excess_return_vs_spy_30d: float
    days_in_probation: int = 0


@router.get("/status")
async def status(service: StrategyService = Depends(get_service)) -> dict:
    """Strategy name + current decay status."""
    return {"strategy": service.name, "status": service.health.status}


@router.post("/evaluate/{symbol}")
async def evaluate(
    symbol: str,
    interval: Interval = Interval.D1,
    service: StrategyService = Depends(get_service),
) -> dict:
    """Manually evaluate a symbol: fetch features → generate + risk-check a signal."""
    try:
        event = await service.evaluate_symbol(symbol.upper(), interval)
    except httpx.HTTPError as exc:
        logger.error("feature-engine query failed", symbol=symbol, error=str(exc))
        raise HTTPException(status_code=502, detail=f"feature-engine query failed: {exc}") from exc
    if event is None:
        return {"symbol": symbol.upper(), "signal": None}
    return {
        "symbol": event.symbol,
        "signal": event.signal,
        "confidence": event.confidence,
        "price": event.price,
        "stop_loss": event.stop_loss,
        "take_profit": event.take_profit,
    }


@router.post("/decay")
async def decay(
    metrics: DecayMetrics,
    service: StrategyService = Depends(get_service),
) -> dict:
    """Re-evaluate strategy health from performance metrics (StrategyDecayMonitor)."""
    event = await service.update_health(**metrics.model_dump())
    return {
        "strategy": service.name,
        "status": service.health.status,
        "status_changed": event is not None,
    }
