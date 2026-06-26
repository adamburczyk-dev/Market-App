import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from trading_common.events import SignalGeneratedEvent

from src.api.deps import get_service
from src.core.service import RiskMgmtService

logger = structlog.get_logger()
router = APIRouter()


class PortfolioUpdate(BaseModel):
    value: float | None = None
    exposure_pct: float | None = None
    drawdown_pct: float | None = None
    daily_loss_pct: float | None = None
    regime: str | None = None


class SignalInput(BaseModel):
    symbol: str
    signal: str  # "BUY" | "SELL" | "HOLD"
    price: float
    confidence: float = 0.8
    stop_loss: float | None = None
    take_profit: float | None = None
    strategy_name: str = "manual"


def _breaker_view(service: RiskMgmtService) -> dict:
    level = service.breaker.level
    return {"level": level.value if level else None, "tripped": service.breaker.is_tripped}


@router.get("/portfolio")
async def get_portfolio(service: RiskMgmtService = Depends(get_service)) -> dict:
    return service.portfolio.as_dict()


@router.post("/portfolio")
async def update_portfolio(
    body: PortfolioUpdate,
    service: RiskMgmtService = Depends(get_service),
) -> dict:
    """Update portfolio state; re-arms the circuit breaker."""
    event = await service.update_portfolio(
        value=body.value,
        exposure_pct=body.exposure_pct,
        drawdown_pct=body.drawdown_pct,
        daily_loss_pct=body.daily_loss_pct,
        regime=body.regime,
    )
    return {
        "portfolio": service.portfolio.as_dict(),
        "circuit_breaker": _breaker_view(service),
        "breaker_changed": event is not None,
    }


@router.get("/circuit-breaker")
async def circuit_breaker(service: RiskMgmtService = Depends(get_service)) -> dict:
    return _breaker_view(service)


@router.post("/signal")
async def process_signal(
    body: SignalInput,
    service: RiskMgmtService = Depends(get_service),
) -> dict:
    """Manually size a signal into an order (the NATS path does this on events)."""
    signal = SignalGeneratedEvent(
        symbol=body.symbol,
        strategy_name=body.strategy_name,
        signal=body.signal,
        confidence=body.confidence,
        price=body.price,
        stop_loss=body.stop_loss,
        take_profit=body.take_profit,
    )
    order = await service.process_signal(signal)
    if order is None:
        return {"order": None}
    return {
        "order": {
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": order.price,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
        }
    }
