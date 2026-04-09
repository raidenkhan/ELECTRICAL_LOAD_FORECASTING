from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.forecast_service import ForecastService
from app.api.deps import get_database

router = APIRouter()
forecast_service = ForecastService()


@router.get("/status")
async def get_model_status():
    """
    Placeholder for model status endpoint.
    Will be implemented in Stage 6.
    """
    return {"message": "Model endpoints - Coming in Stage 6"}


@router.get("/metrics")
async def get_model_metrics(db: AsyncSession = Depends(get_database)):
    """
    Get performance metrics for the forecasting models including trends and heatmaps.
    """
    try:
        return await forecast_service.get_performance_metrics(db)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
