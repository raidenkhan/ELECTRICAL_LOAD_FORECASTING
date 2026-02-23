from fastapi import APIRouter, HTTPException, status
from typing import Any
from app.services.forecast_service import ForecastService
from app.schemas.forecast import ShapResponse

router = APIRouter()
forecast_service = ForecastService()


@router.get("/status")
async def get_status():
    """
    Placeholder for explainability endpoints.
    Will be implemented in Stage 6.
    """
    return {"message": "Explainability endpoints - Coming in Stage 6"}


@router.get("/shap", response_model=ShapResponse)
async def get_shap():
    """
    Get SHAP values for the latest STLF forecast.
    """
    try:
        result = await forecast_service.get_shap_values()
        return ShapResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
