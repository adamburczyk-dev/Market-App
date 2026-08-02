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
    """Every rule this instance runs, with its decay status and declared inputs."""
    return {"strategies": service.statuses()}


@router.post("/evaluate/{symbol}")
async def evaluate(
    symbol: str,
    interval: Interval = Interval.D1,
    service: StrategyService = Depends(get_service),
) -> dict:
    """Manually evaluate a symbol: fetch features → one signal per ACTIVE rule.

    Returns a list, not a signal: rules legitimately disagree, and collapsing
    that here would hide exactly what the aggregator exists to weigh.
    """
    try:
        events = await service.evaluate_symbol(symbol.upper(), interval)
    except httpx.HTTPError as exc:
        logger.error("feature-engine query failed", symbol=symbol, error=str(exc))
        raise HTTPException(status_code=502, detail=f"feature-engine query failed: {exc}") from exc
    return {
        "symbol": symbol.upper(),
        "signals": [
            {
                "strategy": event.strategy_name,
                "signal": event.signal,
                "confidence": event.confidence,
                "price": event.price,
                "stop_loss": event.stop_loss,
                "take_profit": event.take_profit,
            }
            for event in events
        ],
    }


@router.post("/decay/{strategy_name}")
async def decay(
    strategy_name: str,
    metrics: DecayMetrics,
    service: StrategyService = Depends(get_service),
) -> dict:
    """Re-evaluate ONE strategy's health from its metrics (StrategyDecayMonitor)."""
    try:
        event = await service.update_health(strategy_name, **metrics.model_dump())
        status_now = service.health_of(strategy_name).status
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "strategy": strategy_name,
        "status": status_now,
        "status_changed": event is not None,
    }
