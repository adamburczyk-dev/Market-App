from fastapi import APIRouter

from .routes import router as backtest_router

router = APIRouter()
router.include_router(backtest_router, prefix="/backtest", tags=["backtest"])
