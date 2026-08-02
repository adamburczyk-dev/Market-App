"""Backtest HTTP API — run backtests and walk-forward revalidation on demand."""

from contextlib import contextmanager

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from trading_common.constants import MAX_OHLCV_LIMIT
from trading_common.schemas import Interval
from trading_common.strategies import strategy_names

from src.api.deps import get_service
from src.core.rule_engine import CrossSectionalRuleError
from src.core.service import BacktestService

logger = structlog.get_logger()
router = APIRouter()


@contextmanager
def _named_errors():
    """Turn the two ways a strategy request can be un-runnable into 4xx.

    Both used to be invisible: an unknown name silently produced the built-in
    engine's numbers under whatever label was passed, and a cross-sectional rule
    had no way to say it needs a universe. A 500 would be no better — the caller
    has to learn WHICH name and WHY.
    """
    try:
        yield
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"unknown strategy (known: {strategy_names()})",
        ) from exc
    except CrossSectionalRuleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        # Name the upstream, its status and its URL. Letting this escape gives
        # Starlette's plain-text 500 with no body — the failure mode that once
        # made six different ml-pipeline errors look identical as `HTTP 500: {}`.
        raise HTTPException(
            status_code=502,
            detail=(f"upstream {exc.response.status_code} from {exc.request.url}"),
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"market-data unreachable: {type(exc).__name__}"
        ) from exc


class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    interval: Interval = Interval.D1
    # Shared ceiling: this limit is forwarded verbatim to market-data, so a
    # value this service accepts and that one refuses is a guaranteed 502.
    limit: int = Field(default=500, ge=10, le=MAX_OHLCV_LIMIT)
    params: dict[str, float] | None = None


class RevalidateRequest(BaseModel):
    strategy_name: str
    symbol: str
    original_oos_sharpe: float
    interval: Interval = Interval.D1
    limit: int = Field(default=500, ge=10, le=MAX_OHLCV_LIMIT)
    params: dict[str, float] | None = None


@router.get("/status")
async def status() -> dict:
    return {"service": "backtest", "status": "ready"}


@router.post("/run")
async def run_backtest(
    req: BacktestRequest,
    service: BacktestService = Depends(get_service),
) -> dict:
    """Backtest the named registered rule; publishes BacktestCompletedEvent."""
    with _named_errors():
        result = await service.run_backtest(
            req.strategy_name, req.symbol, req.interval, limit=req.limit, params=req.params
        )
    return {
        "strategy_name": req.strategy_name,
        "symbol": req.symbol,
        **result.as_dict(),
        "equity_curve": downsample(result.equity_curve),
    }


# A chart cannot show more points than it has pixels, and a 20-year run is
# 5000 of them. Downsampling here rather than in the browser keeps the payload
# proportional to what is being asked for.
MAX_CURVE_POINTS = 500


def downsample(curve: list[float], max_points: int = MAX_CURVE_POINTS) -> list[float]:
    """Evenly thin a series, ALWAYS keeping the first and last point.

    The last point is the total return the caller is also being told as a
    scalar; dropping it would let the chart and the number disagree.
    """
    if len(curve) <= max_points:
        return curve
    stride = len(curve) / max_points
    sampled = [curve[int(i * stride)] for i in range(max_points)]
    if sampled[-1] != curve[-1]:
        sampled[-1] = curve[-1]
    return sampled


@router.post("/revalidate")
async def revalidate(
    req: RevalidateRequest,
    service: BacktestService = Depends(get_service),
) -> dict:
    """Walk-forward revalidation; publishes StrategyRevalidatedEvent with a recommendation."""
    with _named_errors():
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
