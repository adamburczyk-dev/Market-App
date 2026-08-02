import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from trading_common.events import OrderRequestedEvent

from src.api.deps import get_service
from src.core.service import ExecutionService

logger = structlog.get_logger()
router = APIRouter()

# Read ceiling for the equity series. Matches the broker's retention default so
# a caller cannot ask for a window the broker was never going to keep.
MAX_EQUITY_POINTS = 2000


class OrderInput(BaseModel):
    symbol: str
    side: str  # "BUY" | "SELL"
    quantity: float
    price: float
    strategy_name: str = "manual"
    stop_loss: float | None = None
    take_profit: float | None = None


@router.get("/portfolio")
async def portfolio(service: ExecutionService = Depends(get_service)) -> dict:
    broker = service.broker
    metrics = broker.metrics()
    return {
        "cash": broker.cash,
        "equity": broker.equity,
        "exposure_pct": metrics["exposure_pct"],
        "drawdown_pct": metrics["drawdown_pct"],
        "daily_loss_pct": metrics["daily_loss_pct"],
    }


@router.get("/positions")
async def positions(service: ExecutionService = Depends(get_service)) -> dict:
    return {"positions": service.broker.positions()}


@router.get("/equity")
async def equity(
    limit: int = Query(default=500, ge=2, le=MAX_EQUITY_POINTS),
    service: ExecutionService = Depends(get_service),
) -> dict:
    """Realized equity, one point per session — the series nothing kept before.

    Deliberately raw: execution owns the broker's history, not the risk
    semantics. VaR, drawdown and the rest are computed by the consumer from
    `trading_common.risk_metrics`, so there is one definition of each.
    """
    points = service.broker.equity_curve(limit=limit)
    return {"points": points, "count": len(points)}


@router.post("/execute")
async def execute(
    body: OrderInput,
    service: ExecutionService = Depends(get_service),
) -> dict:
    """Manually paper-fill an order (the NATS path does this on order.requested)."""
    order = OrderRequestedEvent(
        symbol=body.symbol,
        side=body.side,
        quantity=body.quantity,
        price=body.price,
        strategy_name=body.strategy_name,
        stop_loss=body.stop_loss,
        take_profit=body.take_profit,
    )
    fill = await service.execute(order)
    if fill is None:
        raise HTTPException(
            status_code=409, detail="duplicate order or long-only SELL without a position"
        )
    return {
        "order_id": fill.order_id,
        "symbol": fill.symbol,
        "filled_quantity": fill.filled_quantity,
        "filled_price": fill.filled_price,
    }
