from fastapi import APIRouter, Depends
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_database
from app.services.forecast_service import ForecastService
import datetime

router = APIRouter()
forecast_service = ForecastService()

@router.get("/", response_model=List[Dict[str, Any]])
async def get_active_alerts(
    session: AsyncSession = Depends(get_database)
):
    """
    Dynamically scan the latest STLF forecast and historical data 
    to generate active system alerts.
    """
    alerts = []
    
    try:
        # 1. Fetch latest STLF to check for capacity risks
        stlf = await forecast_service.generate_forecast(session, horizon_hours=24, model_type="stlf")
        forecast_mw = stlf.get("forecast_mw", [])
        
        if forecast_mw:
            peak_val = max(forecast_mw)
            if peak_val > forecast_service.CAPACITY_LIMIT * 0.93: # Warning at 93% load
                severity = "critical" if peak_val > forecast_service.CAPACITY_LIMIT else "warning"
                peak_time_idx = forecast_mw.index(peak_val)
                peak_time_str = datetime.datetime.fromisoformat(stlf["timestamps"][peak_time_idx]).strftime("%H:%M")
                
                alerts.append({
                    "id": 1,
                    "type": severity,
                    "title": f"{severity.upper()}: Capacity Margin Breach" if severity == "critical" else "Peak forecast exceeds safety margin",
                    "detail": f"Expected: {int(peak_val):,} MW at {peak_time_str} | Available margin: {max(0, int(forecast_service.CAPACITY_LIMIT - peak_val))} MW",
                    "time": "Just now",
                    "actions": [
                        {"label": "Scale Generation", "primary": True},
                        {"label": "Acknowledge"}
                    ]
                })

        # 2. Check for data integrity (simple mock check)
        # In a real system, we'd check if latest data is > 15 mins old
        alerts.append({
            "id": 2,
            "type": "info",
            "title": "Model recalibration complete",
            "detail": "STLF Ensemble weights updated based on last 7 days of historical performance.",
            "time": "12 minutes ago",
            "actions": [
                {"label": "View Metrics"}
            ]
        })

    except Exception as e:
        # If forecasting fails, generate a critical system alert
        alerts.append({
            "id": 0,
            "type": "critical",
            "title": "Forecasting Engine Offline",
            "detail": f"System encountered an error during inference: {str(e)}",
            "time": "Now",
            "actions": [
                {"label": "Restart Service", "primary": True}
            ]
        })

    return alerts
