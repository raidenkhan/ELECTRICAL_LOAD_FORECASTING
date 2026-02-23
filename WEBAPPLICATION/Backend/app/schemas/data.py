from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class DataUploadResponse(BaseModel):
    """Response for data upload."""
    upload_id: int
    filename: str
    file_size_bytes: int
    row_count: int
    status: str
    message: str
    

class ValidationSummary(BaseModel):
    """Summary of validation checks."""
    check_name: str
    passed: bool
    details: Dict[str, Any]
    

class ValidationReportResponse(BaseModel):
    """Validation report response."""
    report_id: int
    upload_id: int
    created_at: datetime
    total_rows: int
    valid_rows: int
    invalid_rows: int
    anomaly_count: int
    passed: bool
    validation_checks: list[ValidationSummary]
    error_messages: Optional[str] = None
    
    class Config:
        from_attributes = True


class DataQualityMetrics(BaseModel):
    """Data quality metrics."""
    missing_percentage: float
    anomaly_percentage: float
    net_imbalance_mw: float
    voltage_stability_score: float
    frequency_deviation_hz: float
    

class DataUploadRequest(BaseModel):
    """Request schema for data upload (metadata)."""
    description: Optional[str] = Field(None, description="Optional description of the data")
    source: Optional[str] = Field(None, description="Data source identifier")
