"""signal-aggregator HTTP API — combine component signals into one decision."""

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.deps import get_service
from src.core.aggregator import SignalComponent
from src.core.service import SignalAggregatorService

logger = structlog.get_logger()
router = APIRouter()


class ComponentBody(BaseModel):
    source: str
    signal: str  # "BUY" | "SELL" | "HOLD"
    confidence: float = Field(ge=0.0, le=1.0)


class AggregateRequest(BaseModel):
    symbol: str
    components: list[ComponentBody]
    expected_return_bps: float | None = None
    market_cap_tier: str = "large"
    # optional order-driving context, attached to an actionable aggregate
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    strategy_name: str | None = None
    sector: str | None = None


class OutcomeRequest(BaseModel):
    source: str
    daily_return: float


@router.get("/status")
async def status() -> dict:
    return {"service": "signal-aggregator", "status": "ready"}


@router.get("/weights")
async def weights(service: SignalAggregatorService = Depends(get_service)) -> dict:
    return {"weights": service.weights()}


@router.post("/aggregate")
async def aggregate(
    req: AggregateRequest, service: SignalAggregatorService = Depends(get_service)
) -> dict:
    """Manually combine the *posted* components; publishes SignalAggregatedEvent.

    Ops/testing path ONLY (R9): it aggregates exactly what the caller posts —
    it does NOT read the live per-symbol buffer, does NOT add the current macro
    bias, and does NOT enrich the sector. The live decision path is the
    event-driven one (signal.generated / macro.regime_changed subscribers);
    note the published event still reaches risk-mgmt, so a posted BUY with
    levels WILL be sized into a real (paper) order.
    """
    components = [SignalComponent(c.source, c.signal, c.confidence) for c in req.components]
    result = await service.aggregate(
        req.symbol,
        components,
        expected_return_bps=req.expected_return_bps,
        market_cap_tier=req.market_cap_tier,
        price=req.price,
        stop_loss=req.stop_loss,
        take_profit=req.take_profit,
        strategy_name=req.strategy_name,
        sector=req.sector,
    )
    return {
        "symbol": result.symbol,
        "final_signal": result.final_signal,
        "confidence": result.confidence,
        "score": result.score,
        "components_count": result.components_count,
        "weights": result.weights,
        "cost_filtered": result.cost_filtered,
        "price": result.price,
        "stop_loss": result.stop_loss,
        "take_profit": result.take_profit,
        "strategy_name": result.strategy_name,
        "sector": result.sector,
    }


@router.post("/outcomes")
async def record_outcome(
    req: OutcomeRequest, service: SignalAggregatorService = Depends(get_service)
) -> dict:
    """Record a source's realized daily return → adapts its weight."""
    service.record_outcome(req.source, req.daily_return)
    return {"recorded": True, "weights": service.weights()}
