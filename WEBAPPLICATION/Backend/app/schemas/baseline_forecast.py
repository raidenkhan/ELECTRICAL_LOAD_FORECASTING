from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import date


class BaselineForecastResponse(BaseModel):
    forecast_date: str
    forecast_mw: List[float]
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


class Baseline7DayResponse(BaseModel):
    forecast_date: str
    hourly_mw: List[float]
    daily_aggregates: List[DailyAggregate]


class Baseline30DayResponse(BaseModel):
    forecast_date: str
    daily_aggregates: List[DailyAggregate]


class Baseline90DayResponse(BaseModel):
    forecast_date: str
    weekly_aggregates: List[WeeklyAggregate]


class DataFreshnessInfo(BaseModel):
    latest_date: str
    days_stale: int
    status: str  # "fresh", "stale", "unknown"


class BaselineUploadResponse(BaseModel):
    status: str
    records_loaded: int
    latest_date: str
    days_loaded: int
    forecast: BaselineForecastResponse
    freshness: DataFreshnessInfo
