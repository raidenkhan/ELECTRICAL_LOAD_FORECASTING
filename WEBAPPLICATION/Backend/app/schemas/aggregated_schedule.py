from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel


class HourlyAggregation(BaseModel):
    hour: int
    ecg_forecast_mw: float
    nedco_mw: float
    valco_mw: float
    mines_mw: float
    export_mw: float
    total_demand_mw: float
    total_supply_mw: float
    reserve_mw: float
    reserve_pct: float


class AggregatedScheduleResponse(BaseModel):
    schedule_id: int
    schedule_date: date
    status: str
    source_filename: str
    operator_notes: Optional[str] = None
    hourly: list[HourlyAggregation]
    peak_demand_mw: float
    peak_demand_hour: int
    total_energy_demand_mwh: float
    total_energy_supply_mwh: float
    avg_demand_mw: float
    avg_supply_mw: float
    min_reserve_mw: float
    min_reserve_hour: int
    using_forecast: bool
    computed_at: datetime
