from fastapi import APIRouter

from .routes import router as macro_router

router = APIRouter()
router.include_router(macro_router, prefix="/macro-data", tags=["macro-data"])
