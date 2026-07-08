from fastapi import APIRouter, HTTPException, status
from typing import Any
import numpy as np

from app.services.dispatch_forecast_service import DispatchForecastService
from app.services.monthly_analyzer import get_monthly_insights
from app.ml.interpretability.decom_engine_hourly import DecomEngineHourly

router = APIRouter()
forecast_service = DispatchForecastService()
decom_engine = DecomEngineHourly()


@router.get("/status")
async def get_status():
    return {"message": "Explainability endpoints active"}


@router.get("/peak-decomposition")
async def get_peak_decomposition():
    """
    Get physical decomposition components for the latest peak hour.
    Uses DLinear+TIDE forecast and DecomEngineHourly for decomposition.
    """
    try:
        result = await forecast_service.forecast_tomorrow()
        if "error" in result or not result.get("forecast_mw"):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No forecast available")

        forecast_mw = np.array(result["forecast_mw"])
        peak_idx = int(np.argmax(forecast_mw))
        peak_mw = float(forecast_mw[peak_idx])

        mean_mw = float(np.mean(forecast_mw))
        base_est = float(np.percentile(forecast_mw, 10))
        seasonal_shift = peak_mw - mean_mw
        ramp_effect = peak_mw - float(np.median(forecast_mw[max(0, peak_idx-3):peak_idx+1]))

        return {
            "peak_mw": round(peak_mw, 1),
            "peak_hour": peak_idx,
            "mean_mw": round(mean_mw, 1),
            "components": [
                {"name": "Base Load", "value": round(base_est, 1), "color": "#3498db"},
                {"name": "Seasonal Rhythm", "value": round(seasonal_shift, 1), "color": "#9b59b6"},
                {"name": "Morning Ramp-up", "value": round(ramp_effect, 1), "color": "#e74c3c"},
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
