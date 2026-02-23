from fastapi import APIRouter, HTTPException, status
from typing import List, Any
from app.services.forecast_service import ForecastService
from app.schemas.forecast import MetricResponse

router = APIRouter()
forecast_service = ForecastService()


@router.get("/status")
async def get_model_status():
    """
    Placeholder for model status endpoint.
    Will be implemented in Stage 6.
    """
    return {"message": "Model endpoints - Coming in Stage 6"}


@router.get("/metrics", response_model=List[MetricResponse])
async def get_model_metrics():
    """
    Get performance metrics for the forecasting models.
    """
    try:
        result = await forecast_service.get_performance_metrics()
        return [MetricResponse(**m) for m in result]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
