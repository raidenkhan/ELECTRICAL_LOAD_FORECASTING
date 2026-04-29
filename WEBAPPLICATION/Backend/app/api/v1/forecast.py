
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any

from app.api.deps import get_db
from app.services.forecast_service import ForecastService
from app.schemas.forecast import ForecastRequest, ForecastResponse, SimulationRequest

router = APIRouter()
forecast_service = ForecastService()

@router.post("/stlf", response_model=ForecastResponse)
async def generate_stlf(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Generate Short-Term Load Forecast (STLF).
    Default horizon: 24 hours.
    Uses Ensemble of Autoformer and LightGBM with Kalman Fusion.
    """
    try:
        # Enforce STLF type
        result = await forecast_service.generate_forecast(
            session=db,
            horizon_hours=request.horizon_hours,
            model_type="stlf"
        )
        
        # Construct response
        return ForecastResponse(
            forecast_id="stlf_" + result["timestamps"][0].strftime("%Y%m%d%H%M"), # Simple ID generation
            timestamp=result["timestamps"][0],
            horizon_hours=request.horizon_hours,
            model_type="stlf",
            timestamps=result["timestamps"],
            forecast_mw=result["forecast_mw"],
            simday_forecast_mw=result.get("simday_forecast_mw"),
            p10=result.get("p10"),
            p90=result.get("p90"),
            regime_distribution=result.get("regime_distribution"),
            metadata=result.get("metadata")
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/ltlf", response_model=ForecastResponse)
async def generate_ltlf(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Generate Long-Term Load Forecast (LTLF).
    Default horizon: 30 days.
    Uses Recursive LightGBM Quantile Regressors.
    """
    try:
        # Enforce LTLF type
        result = await forecast_service.generate_forecast(
            session=db,
            horizon_hours=request.horizon_hours,
            model_type="ltlf"
        )
        
        return ForecastResponse(
            forecast_id="ltlf_" + result["timestamps"][0].strftime("%Y%m%d"),
            timestamp=result["timestamps"][0],
            horizon_hours=request.horizon_hours,
            model_type="ltlf",
            timestamps=result["timestamps"],
            forecast_mw=result["forecast_mw"],
            p10=result.get("p10"),
            p90=result.get("p90"),
            metadata=result.get("metadata")
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/simulate", response_model=ForecastResponse)
async def generate_simulation(
    request: SimulationRequest,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Generate a simulation forecast based on what-if parameters.
    """
    try:
        result = await forecast_service.run_simulation(
            session=db,
            horizon_hours=request.horizon_hours,
            temp_offset=request.temp_offset,
            inflow_offset_pct=request.inflow_offset_pct,
            industrial_load_offset_pct=request.industrial_load_offset_pct
        )
        
        return ForecastResponse(
            forecast_id="sim_" + result["timestamps"][0].strftime("%Y%m%d%H%M"),
            timestamp=result["timestamps"][0],
            horizon_hours=request.horizon_hours,
            model_type="simulation",
            timestamps=result["timestamps"],
            forecast_mw=result["forecast_mw"],
            p10=result.get("p10"),
            p90=result.get("p90"),
            metadata={
                **result.get("metadata", {}),
                "offsets": result.get("offsets")
            }
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
