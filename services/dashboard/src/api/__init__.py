from fastapi import APIRouter

from .routes import router as dashboard_router

router = APIRouter()
router.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
