from fastapi import APIRouter, HTTPException, status
from typing import Any

from app.services.dispatch_forecast_service import DispatchForecastService
from app.services.baseline_forecast_service import BaselineForecastService
from app.db.session import AsyncSessionLocal
from app.db.models.forecast_metrics import ForecastMetrics
from sqlalchemy import select, func

router = APIRouter()
dispatch_service = DispatchForecastService()


@router.get("/status")
async def get_model_status():
    return {"message": "Model endpoints active"}


@router.get("/metrics")
async def get_model_metrics():
    """
    Get live performance metrics from both forecasting engines.
    """
    try:
        # Get rolling MAE from ForecastMetrics table
        async with AsyncSessionLocal() as db:
            stmt = (
                select(
                    ForecastMetrics.horizon,
                    func.avg(ForecastMetrics.mae),
                    func.avg(ForecastMetrics.mape),
                    func.count(ForecastMetrics.id),
                )
                .group_by(ForecastMetrics.horizon)
                .order_by(ForecastMetrics.horizon)
            )
            result = await db.execute(stmt)
            rows = result.all()

        db_metrics = [
            {
                "horizon": r[0],
                "mae": round(float(r[1]), 1) if r[1] else None,
                "mape": round(float(r[2]), 2) if r[2] else None,
                "n_samples": r[3],
            }
            for r in rows
        ] if rows else []

        # Get real-time engine health
        health = {}
        if dispatch_service.engine:
            health["dlinear_tide"] = dispatch_service.engine.health()
        if dispatch_service.engine and dispatch_service.engine.is_fitted:
            health["rolling_mae_24h"] = dispatch_service.engine.health().get("mae_24h")

        return {
            "db_metrics": db_metrics,
            "engine_health": health,
            "note": "Metrics from ForecastMetrics table (DB) and live engine health.",
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
