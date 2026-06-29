"""Backtest HTTP API — run backtests and walk-forward revalidation on demand."""

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from trading_common.schemas import Interval

from src.api.deps import get_service
from src.core.service import BacktestService

logger = structlog.get_logger()
router = APIRouter()


class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    interval: Interval = Interval.D1
    limit: int = Field(default=500, ge=10, le=5000)
    params: dict[str, float] | None = None


class RevalidateRequest(BaseModel):
    strategy_name: str
    symbol: str
    original_oos_sharpe: float
    interval: Interval = Interval.D1
    limit: int = Field(default=500, ge=10, le=5000)
    params: dict[str, float] | None = None


@router.get("/status")
async def status() -> dict:
    return {"service": "backtest", "status": "ready"}


@router.post("/run")
async def run_backtest(
    req: BacktestRequest,
    service: BacktestService = Depends(get_service),
) -> dict:
    """Run a full-sample momentum backtest and publish BacktestCompletedEvent."""
    result = await service.run_backtest(
        req.strategy_name, req.symbol, req.interval, limit=req.limit, params=req.params
    )
    return {"strategy_name": req.strategy_name, "symbol": req.symbol, **result.as_dict()}


@router.post("/revalidate")
async def revalidate(
    req: RevalidateRequest,
    service: BacktestService = Depends(get_service),
) -> dict:
    """Walk-forward revalidation; publishes StrategyRevalidatedEvent with a recommendation."""
    result = await service.revalidate(
        req.strategy_name,
        req.symbol,
        req.original_oos_sharpe,
        req.interval,
        limit=req.limit,
        params=req.params,
    )
    return {
        "strategy_name": result.strategy_name,
        "original_oos_sharpe": result.original_oos_sharpe,
        "current_oos_sharpe": result.current_oos_sharpe,
        "degradation_pct": result.degradation_pct,
        "recommended_status": result.recommended_status,
        "oos_window_days": result.oos_window_days,
        "is_window_days": result.is_window_days,
    }
