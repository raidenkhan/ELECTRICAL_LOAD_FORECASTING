from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import date


class DispatchForecastRequest(BaseModel):
    target_date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format. Defaults to tomorrow.")
    temperature_c: Optional[List[float]] = Field(None, min_length=24, max_length=24, description="24 hourly temperature values in °C. If omitted, uses Open-Meteo forecast or seasonal fallback.")


class DispatchForecastResponse(BaseModel):
    forecast_date: str
    forecast_mw: List[float]
    p10_mw: Optional[List[float]] = None
    p90_mw: Optional[List[float]] = None
    uncertainty_mw: Optional[List[float]] = None
    temperature_c: Optional[List[float]] = None
    components: Optional[Dict[str, Any]] = None
    factors: Optional[Dict[str, Any]] = None


class DailyAggregate(BaseModel):
    date: str
    peak_mw: float
    mean_mw: float
    min_mw: float
    total_energy_mwh: float


class WeeklyAggregate(BaseModel):
    week_start: str
    week_end: str
    mean_mw: float
    peak_mw: float
    min_mw: float
    total_energy_mwh: float


class Forecast7DayResponse(BaseModel):
    forecast_date: str
    hourly_mw: List[float]
    p10_mw: Optional[List[float]] = None
    p90_mw: Optional[List[float]] = None
    uncertainty_mw: Optional[List[float]] = None
    temperature_c: Optional[List[float]] = None
    daily_aggregates: List[DailyAggregate]


class TemperatureOverrideRequest(BaseModel):
    temperature_c: List[float] = Field(..., min_length=24, max_length=24, description="24 hourly temperature values in °C")


class Forecast30DayResponse(BaseModel):
    forecast_date: str
    daily_aggregates: List[DailyAggregate]


class Forecast90DayResponse(BaseModel):
    forecast_date: str
    weekly_aggregates: List[WeeklyAggregate]


class FeedbackRequest(BaseModel):
    actual_mw: List[float] = Field(..., min_length=24, max_length=24, description="24 hourly actual demand values")
    forecast_mw: Optional[List[float]] = Field(None, min_length=24, max_length=24, description="24 hourly forecast values. If omitted, looked up from cache.")
    forecast_date: Optional[str] = Field(None, description="Date the forecast was for (YYYY-MM-DD). Defaults to today.")
