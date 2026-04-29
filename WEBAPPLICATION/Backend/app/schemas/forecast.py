
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class RegimeBin(BaseModel):
    hour: str
    regime0: float  # Standard (%)
    regime1: float  # Transition (%)
    regime2: float  # Peak (%)

class ForecastRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    horizon_hours: int = Field(24, ge=1, le=720, description="Forecast horizon in hours")
    model_type: str = Field("stlf", pattern="^(stlf|ltlf)$", description="Model type: stlf (Short-Term) or ltlf (Long-Term)")

class SimulationRequest(BaseModel):
    horizon_hours: int = Field(24, ge=1, le=720)
    temp_offset: float = Field(0.0, description="Temperature offset in Celsius")
    inflow_offset_pct: float = Field(0.0, description="Grid inflow adjustment in percentage")
    industrial_load_offset_pct: float = Field(0.0, description="Industrial load adjustment in percentage")

class ShapResponse(BaseModel):
    features: List[str]
    values: List[float]
    base_value: float

class MetricResponse(BaseModel):
    mae: float
    rmse: float
    mape: float
    horizon: str
    sample_count: int

class ForecastResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    forecast_id: str
    timestamp: datetime
    horizon_hours: int
    model_type: str
    
    # Time series data
    timestamps: List[datetime]
    forecast_mw: List[float]
    p10: Optional[List[float]] = None
    p90: Optional[List[float]] = None
    regime_distribution: Optional[List[RegimeBin]] = None
    
    # Optional Comparison Model (GRIDCo Similar Day)
    simday_forecast_mw: Optional[List[float]] = None
    
    metadata: Optional[Dict[str, Any]] = None
