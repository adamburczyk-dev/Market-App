import structlog
from fastapi import APIRouter

logger = structlog.get_logger()
router = APIRouter()


@router.get("/status")
async def status() -> dict:
    """Status serwisu — placeholder do implementacji."""
    return {"service": "backtest", "status": "skeleton"}
