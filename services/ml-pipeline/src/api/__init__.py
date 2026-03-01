from fastapi import APIRouter

from .routes import router as ml_pipeline_router

router = APIRouter()
router.include_router(ml_pipeline_router, prefix="/ml-pipeline", tags=["ml-pipeline"])
