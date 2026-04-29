from fastapi import APIRouter, HTTPException, status, Depends
from typing import Any
from sqlalchemy.orm import Session
from app.services.forecast_service import ForecastService
from app.schemas.forecast import ShapResponse
from app.api.deps import get_database

router = APIRouter()
forecast_service = ForecastService()


@router.get("/status")
async def get_status():
    """
    Placeholder for explainability endpoints.
    Will be implemented in Stage 6.
    """
    return {"message": "Explainability endpoints - Coming in Stage 6"}


@router.get("/peak-decomposition")
async def get_peak_decomposition(db: Session = Depends(get_database)):
    """
    Get physical decomposition components for the latest peak hour.
    """
    try:
        # Get the latest STLF forecast from cache or generate it
        forecast = await forecast_service.generate_forecast(db, horizon_hours=24, model_type="stlf")
        result = await forecast_service.get_peak_decomposition(forecast)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
