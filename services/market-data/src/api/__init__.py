from fastapi import APIRouter
from .routes import router as ohlcv_router

router = APIRouter()
router.include_router(ohlcv_router, prefix="/market-data", tags=["market-data"])
