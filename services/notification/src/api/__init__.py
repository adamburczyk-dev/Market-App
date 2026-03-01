from fastapi import APIRouter
from .routes import router as notification_router

router = APIRouter()
router.include_router(notification_router, prefix="/notification", tags=["notification"])
