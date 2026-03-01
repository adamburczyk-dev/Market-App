from fastapi import APIRouter

from .routes import router as execution_router

router = APIRouter()
router.include_router(execution_router, prefix="/execution", tags=["execution"])
