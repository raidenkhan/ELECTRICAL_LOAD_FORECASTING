from fastapi import APIRouter, HTTPException, status, Query
from typing import Any, Optional, List
from datetime import date, timedelta

from app.services.dispatch_forecast_service import DispatchForecastService
from app.services.baseline_forecast_service import BaselineForecastService
from app.schemas.dispatch_forecast import (
    DispatchForecastRequest,
    DispatchForecastResponse,
    Forecast7DayResponse,
    Forecast30DayResponse,
    Forecast90DayResponse,
    FeedbackRequest,
    TemperatureOverrideRequest,
)

router = APIRouter()
forecast_service = DispatchForecastService()
baseline_service = BaselineForecastService()


@router.post("/dispatch", response_model=DispatchForecastResponse)
async def forecast_dispatch(request: DispatchForecastRequest) -> Any:
    if not forecast_service.engine or not forecast_service.engine.is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dispatch forecast engine not trained. Run train_ecg_engine.py first.",
        )
    try:
        result = await forecast_service.forecast_dispatch(request.target_date, user_temps=request.temperature_c)
        if "error" in result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["error"])
        return DispatchForecastResponse(
            forecast_date=result["forecast_date"],
            forecast_mw=result["forecast_mw"],
            p10_mw=result.get("p10_mw"),
            p90_mw=result.get("p90_mw"),
            uncertainty_mw=result.get("uncertainty_mw"),
            temperature_c=result.get("temperature_c"),
            components=result.get("components"),
            factors=result.get("factors"),
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/dispatch/tomorrow", response_model=DispatchForecastResponse)
async def forecast_tomorrow(force_refresh: bool = Query(False, description="Skip cache and regenerate")) -> Any:
    if not forecast_service.engine or not forecast_service.engine.is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dispatch forecast engine not trained. Run train_ecg_engine.py first.",
        )
    try:
        result = await forecast_service.forecast_tomorrow(force_refresh=force_refresh)
        if "error" in result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["error"])

        return DispatchForecastResponse(
            forecast_date=result["forecast_date"],
            forecast_mw=result["forecast_mw"],
            p10_mw=result.get("p10_mw"),
            p90_mw=result.get("p90_mw"),
            uncertainty_mw=result.get("uncertainty_mw"),
            temperature_c=result.get("temperature_c"),
            components=result.get("components"),
            factors=result.get("factors"),
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/dispatch/7day", response_model=Forecast7DayResponse)
async def forecast_7day(force_refresh: bool = Query(False, description="Skip cache and regenerate")) -> Any:
    if not forecast_service.engine or not forecast_service.engine.is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dispatch forecast engine not trained.",
        )
    try:
        result = await forecast_service.forecast_7day(force_refresh=force_refresh)
        if "error" in result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["error"])
        return Forecast7DayResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/dispatch/30day", response_model=Forecast30DayResponse)
async def forecast_30day(force_refresh: bool = Query(False, description="Skip cache and regenerate")) -> Any:
    if not forecast_service.engine or not forecast_service.engine.is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dispatch forecast engine not trained.",
        )
    try:
        result = await forecast_service.forecast_30day(force_refresh=force_refresh)
        if "error" in result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["error"])
        return Forecast30DayResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/dispatch/90day", response_model=Forecast90DayResponse)
async def forecast_90day(force_refresh: bool = Query(False, description="Skip cache and regenerate")) -> Any:
    if not forecast_service.engine or not forecast_service.engine.is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dispatch forecast engine not trained.",
        )
    try:
        result = await forecast_service.forecast_90day(force_refresh=force_refresh)
        if "error" in result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["error"])
        return Forecast90DayResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/dispatch/refresh")
async def refresh_all_caches():
    """Clear all cached forecasts so next request regenerates."""
    await forecast_service._cache_clear_all()
    await baseline_service._cache_clear_all()
    return {"status": "ok", "message": "Forecast cache cleared"}


@router.get("/dispatch/compare")
async def compare_engines(target_date: Optional[str] = Query(None, description="Date YYYY-MM-DD, defaults to tomorrow")):
    """Compare DLinear+TIDE vs baseline (WT+DOW) forecasts side by side."""
    dt = date.fromisoformat(target_date) if target_date else date.today() + timedelta(days=1)
    dlinear = await forecast_service.forecast_for_date(dt)
    baseline = await baseline_service.forecast_for_date(dt)
    return {
        "comparison_date": dt.isoformat(),
        "dlinear": {
            "forecast_mw": dlinear.get("forecast_mw"),
            "p10_mw": dlinear.get("p10_mw"),
            "p90_mw": dlinear.get("p90_mw"),
            "uncertainty_mw": dlinear.get("uncertainty_mw"),
            "engine": dlinear.get("engine"),
        },
        "baseline": {
            "forecast_mw": baseline.get("forecast_mw"),
            "engine": "weighted_trend_dow",
        },
        "note": "DLinear+TIDE is the primary engine. Baseline (WT+DOW) shown for reference.",
    }


@router.post("/dispatch/feedback")
async def feedback_actuals(body: FeedbackRequest):
    """Feed actual demand back into TIDE corrector.

    Provide 24 hourly actual_mw values and optionally a forecast_date.
    If forecast_mw is not given, the endpoint looks up the cached forecast.
    """
    dt = date.fromisoformat(body.forecast_date) if body.forecast_date else date.today()
    ok = await forecast_service.feedback_actuals(body.actual_mw, body.forecast_mw, dt)
    if not ok:
        raise HTTPException(status_code=400, detail="Feedback failed — check actual_mw length and cached forecast availability")
    return {"status": "ok", "message": f"TIDE updated with actuals for {dt}"}


@router.post("/dispatch/temperature-override")
async def set_temperature_override(body: TemperatureOverrideRequest):
    """Override forecast temperature with manual values (24 hourly °C).
    Useful when Open-Meteo is down or for what-if scenarios.
    Call /dispatch/temperature-clear to revert to Open-Meteo.
    """
    forecast_service.set_manual_temperature(body.temperature_c)
    return {"status": "ok", "message": "Manual temperature override set", "temperature_c": body.temperature_c}


@router.post("/dispatch/temperature-clear")
async def clear_temperature_override():
    """Revert to Open-Meteo temperature forecast."""
    forecast_service.clear_manual_temperature()
    return {"status": "ok", "message": "Temperature override cleared — using Open-Meteo"}


@router.get("/dispatch/current-temp")
async def current_temperature():
    """Get the current estimated temperature for Accra."""
    temp = await forecast_service.weather_service.get_current_temp()
    return {"temperature_c": temp, "location": "Accra, Ghana"}
