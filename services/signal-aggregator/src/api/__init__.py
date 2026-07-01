from fastapi import APIRouter

from .routes import router as aggregator_router

router = APIRouter()
router.include_router(aggregator_router, prefix="/signal-aggregator", tags=["signal-aggregator"])
