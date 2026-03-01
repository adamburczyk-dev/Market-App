from fastapi import APIRouter
from .routes import router as risk_mgmt_router

router = APIRouter()
router.include_router(risk_mgmt_router, prefix="/risk-mgmt", tags=["risk-mgmt"])
