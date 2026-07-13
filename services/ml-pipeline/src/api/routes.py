"""ml-pipeline HTTP API — training, model registry, baselines and drift checks."""

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from trading_common.schemas import Interval

from src.api.deps import get_service
from src.core.service import MLPipelineService

logger = structlog.get_logger()
router = APIRouter()


class TrainRequest(BaseModel):
    symbols: list[str] = Field(min_length=2)  # cross-sectional — needs a universe
    interval: str = "1d"
    limit: int = Field(default=1500, ge=300, le=10_000)


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
    versions = service.model_store.versions() if service.model_store is not None else []
    return {"models": service.registry.model_ids(), "registry_versions": versions}


@router.post("/models/train")
async def train(req: TrainRequest, service: MLPipelineService = Depends(get_service)) -> dict:
    """Run the full training pass (plan §6–§7): dataset → purged walk-forward →
    gate report → MLflow version + drift baseline. Synchronous and potentially
    minutes-long — ops/scheduled use; promotion to production stays manual."""
    try:
        return await service.train(req.symbols, Interval(req.interval), limit=req.limit)
    except RuntimeError as exc:  # market-data client not configured
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:  # dataset too small for holdout + folds
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/versions/{version}/promote")
async def promote(version: str, service: MLPipelineService = Depends(get_service)) -> dict:
    """Point the ``production`` alias at a version (manual gate sign-off)."""
    if service.model_store is None:
        raise HTTPException(status_code=503, detail="model store unavailable")
    service.model_store.promote(version)
    return {"model": service.model_store.model_name, "production_version": version}


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
