"""ml-pipeline HTTP API — register model baselines and run drift checks."""

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.deps import get_service
from src.core.service import MLPipelineService

logger = structlog.get_logger()
router = APIRouter()


class BaselineRequest(BaseModel):
    reference_features: dict[str, list[float]]
    baseline_sharpe: float
    prediction_reference: list[float] = Field(default_factory=list)


class DriftCheckRequest(BaseModel):
    current_features: dict[str, list[float]]
    rolling_sharpe_30d: float
    rolling_sharpe_90d: float
    rolling_accuracy_30d: float
    prediction_current: list[float] = Field(default_factory=list)


@router.get("/status")
async def status() -> dict:
    return {"service": "ml-pipeline", "status": "ready"}


@router.get("/models")
async def list_models(service: MLPipelineService = Depends(get_service)) -> dict:
    return {"models": service.registry.model_ids()}


@router.post("/models/{model_id}/baseline")
async def register_baseline(
    model_id: str,
    req: BaselineRequest,
    service: MLPipelineService = Depends(get_service),
) -> dict:
    """Register a model's training reference distributions + baseline Sharpe."""
    service.register_baseline(
        model_id,
        req.reference_features,
        req.baseline_sharpe,
        req.prediction_reference,
    )
    return {"model_id": model_id, "registered": True, "features": sorted(req.reference_features)}


@router.post("/models/{model_id}/drift")
async def check_drift(
    model_id: str,
    req: DriftCheckRequest,
    service: MLPipelineService = Depends(get_service),
) -> dict:
    """Run the drift check; publishes ModelDriftDetectedEvent when actionable."""
    report = await service.check_drift(
        model_id,
        req.current_features,
        req.rolling_sharpe_30d,
        req.rolling_sharpe_90d,
        req.rolling_accuracy_30d,
        prediction_current=req.prediction_current,
    )
    if report is None:
        raise HTTPException(status_code=404, detail=f"no baseline registered for {model_id}")
    return {
        "model_id": report.model_id,
        "report_date": report.report_date.isoformat(),
        "feature_psi_scores": report.feature_psi_scores,
        "features_drifted": report.features_drifted,
        "sharpe_decay_pct": report.sharpe_decay_pct,
        "needs_retrain": report.needs_retrain,
        "needs_investigation": report.needs_investigation,
        "recommended_action": report.recommended_action,
    }
