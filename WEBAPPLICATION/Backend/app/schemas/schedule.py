from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel


class HourlyDemandItem(BaseModel):
    hour: int
    entity_name: str
    demand_mw: float
    is_forecasted: bool = False


class HourlySupplyItem(BaseModel):
    hour: int
    plant_name: str
    supply_mw: float
    is_baseload: bool = False
    category: str = ""


class ScheduleUploadResponse(BaseModel):
    id: int
    date: date
    status: str
    source_filename: str
    demand_count: int
    supply_count: int
    message: str


class ScheduleDetail(BaseModel):
    id: int
    date: date
    status: str
    source_filename: str
    operator_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    demand: list[HourlyDemandItem]
    supply: list[HourlySupplyItem]

    class Config:
        from_attributes = True


class CellUpdateRequest(BaseModel):
    table: str = "demand"
    entity_name: str
    hour: int
    value: float


class ConfirmRequest(BaseModel):
    operator_notes: Optional[str] = None


class ReviseRequest(BaseModel):
    operator_notes: str
