import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from trading_common.events import OrderRequestedEvent

from src.api.deps import get_service
from src.core.service import ExecutionService

logger = structlog.get_logger()
router = APIRouter()


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
