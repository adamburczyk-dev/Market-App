"""company-classifier HTTP API — classify a profile's style + model stack."""

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from trading_common.schemas import CompanyProfile

from src.api.deps import get_service
from src.core.classifier import ClassificationResult, ValuationMetrics
from src.core.service import CompanyClassifierService

logger = structlog.get_logger()
router = APIRouter()


class MetricsBody(BaseModel):
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    dividend_yield: float | None = None


class ClassifyRequest(BaseModel):
    profile: CompanyProfile
    metrics: MetricsBody | None = None


def _view(profile: CompanyProfile, result: ClassificationResult) -> dict:
    return {
        "profile": profile.model_dump(mode="json"),
        "style": result.style,
        "model_stack": result.model_stack,
        "cap_tier": result.cap_tier,
        "basis": result.basis,
        "scores": {"growth": result.growth_score, "value": result.value_score},
    }


@router.get("/status")
async def status() -> dict:
    return {"service": "company-classifier", "status": "ready"}


@router.get("/companies")
async def list_companies(service: CompanyClassifierService = Depends(get_service)) -> dict:
    return {"symbols": service.symbols()}


@router.get("/companies/{symbol}")
async def get_company(
    symbol: str, service: CompanyClassifierService = Depends(get_service)
) -> dict:
    record = service.get(symbol)
    if record is None:
        raise HTTPException(status_code=404, detail=f"{symbol} not classified")
    return _view(*record)


@router.post("/classify")
async def classify(
    req: ClassifyRequest, service: CompanyClassifierService = Depends(get_service)
) -> dict:
    """Classify style + route the model stack; publishes CompanyClassifiedEvent."""
    metrics = ValuationMetrics(**req.metrics.model_dump()) if req.metrics else None
    record = await service.classify(req.profile, metrics)
    return _view(*record)
