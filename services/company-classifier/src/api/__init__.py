from fastapi import APIRouter

from .routes import router as classifier_router

router = APIRouter()
router.include_router(classifier_router, prefix="/company-classifier", tags=["company-classifier"])
