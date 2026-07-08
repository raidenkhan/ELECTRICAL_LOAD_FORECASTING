from fastapi import APIRouter, HTTPException, status, Query, UploadFile, File
from typing import Any

from datetime import date

from app.services.baseline_forecast_service import BaselineForecastService
from app.services.dispatch_forecast_service import DispatchForecastService
from app.schemas.baseline_forecast import (
    BaselineForecastResponse,
    Baseline7DayResponse,
    Baseline30DayResponse,
    Baseline90DayResponse,#prolly should remove that for 90 day forecast, as it is not very useful and we can save resources by not running it, or we can run it but not return it in the API, just keep it for internal use to track long term trends
    BaselineUploadResponse,
    DataFreshnessInfo,
)

router = APIRouter()
baseline_service = BaselineForecastService()
forecast_service = DispatchForecastService()


@router.get("/baseline/tomorrow", response_model=BaselineForecastResponse)
async def baseline_tomorrow(force_refresh: bool = Query(False, description="Skip cache")) -> Any:
    if baseline_service.engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Baseline engine not initialized")# alerts should be triggered
    try:
        result = await baseline_service.forecast_tomorrow(force_refresh=force_refresh) 
        if "error" in result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["error"])
        return BaselineForecastResponse(
            forecast_date=result["forecast_date"],
            forecast_mw=result["forecast_mw"],
            factors=result.get("factors"),
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/baseline/7day", response_model=Baseline7DayResponse)
async def baseline_7day(force_refresh: bool = Query(False, description="Skip cache")) -> Any:
    try:
        result = await baseline_service.forecast_7day(force_refresh=force_refresh)
        if "error" in result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["error"])
        return Baseline7DayResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/baseline/30day", response_model=Baseline30DayResponse)
async def baseline_30day(force_refresh: bool = Query(False, description="Skip cache")) -> Any:
    try:
        result = await baseline_service.forecast_30day(force_refresh=force_refresh)
        if "error" in result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["error"])
        return Baseline30DayResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/baseline/90day", response_model=Baseline90DayResponse) # this call takes a while, should it really be lumped with the other calls
async def baseline_90day(force_refresh: bool = Query(False, description="Skip cache")) -> Any:
    try:
        result = await baseline_service.forecast_90day(force_refresh=force_refresh)
        if "error" in result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["error"])
        return Baseline90DayResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/baseline/freshness", response_model=DataFreshnessInfo)
async def baseline_freshness() -> Any:
    info = await baseline_service.data_freshness_info()
    return DataFreshnessInfo(**info)


@router.post("/baseline/upload", response_model=BaselineUploadResponse)
async def baseline_upload(file: UploadFile = File(...)) -> Any:
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only CSV files accepted")
    try:
        result = await baseline_service.quick_refit(file)
        if result["status"] == "error":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("detail", "Upload failed"))

        # Feed TIDE with uploaded actuals so bias correction learns
        actual_mw = result.get("_actual_mw")
        if actual_mw:
            dt = date.fromisoformat(result["latest_date"])
            await forecast_service.feedback_actuals(actual_mw, forecast_mw=None, forecast_date=dt)

        return BaselineUploadResponse(
            status=result["status"],
            records_loaded=result["records_loaded"],
            latest_date=result["latest_date"],
            days_loaded=result["days_loaded"],
            forecast=BaselineForecastResponse(
                forecast_date=result["forecast"]["forecast_date"],
                forecast_mw=result["forecast"]["forecast_mw"],
                factors=result["forecast"].get("factors"),
            ),
            freshness=DataFreshnessInfo(**result["freshness"]),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
