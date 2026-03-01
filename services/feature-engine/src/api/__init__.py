from fastapi import APIRouter

from .routes import router as feature_engine_router

router = APIRouter()
router.include_router(feature_engine_router, prefix="/feature-engine", tags=["feature-engine"])
