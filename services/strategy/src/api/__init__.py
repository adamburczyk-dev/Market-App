from fastapi import APIRouter
from .routes import router as strategy_router

router = APIRouter()
router.include_router(strategy_router, prefix="/strategy", tags=["strategy"])
