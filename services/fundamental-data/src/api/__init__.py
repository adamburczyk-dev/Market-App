from fastapi import APIRouter

from .routes import router as fundamentals_router

router = APIRouter()
router.include_router(fundamentals_router, prefix="/fundamental-data", tags=["fundamental-data"])
