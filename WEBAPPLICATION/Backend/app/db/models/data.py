from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON

from app.db.base import Base


class RawDataUpload(Base):
    """Track uploaded CSV files."""
    
    __tablename__ = "raw_data_uploads"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    row_count = Column(Integer)
    status = Column(String(50), default="pending")  # pending, validated, failed
    validation_report_id = Column(Integer, nullable=True)
    

class ValidatedData(Base):
    """Store validated time-series load data."""
    
    __tablename__ = "validated_data"
    
    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Load data (MW)
    total_load_mw = Column(Float, nullable=False)
    line1_mw = Column(Float)
    line2_mw = Column(Float)
    line3_mw = Column(Float)
    
    # Auxiliary variables
    voltage_kv = Column(Float)
    current_a = Column(Float)
    temperature_c = Column(Float)
    frequency_hz = Column(Float)
    
    # Validation flags
    is_anomaly = Column(Boolean, default=False)
    validation_flags = Column(JSON, default={})
    

class ValidationReport(Base):
    """Store validation results for uploaded data."""
    
    __tablename__ = "validation_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Summary statistics
    total_rows = Column(Integer, nullable=False)
    valid_rows = Column(Integer, nullable=False)
    invalid_rows = Column(Integer, nullable=False)
    anomaly_count = Column(Integer, default=0)
    
    # Validation results
    validation_summary = Column(JSON, nullable=False)
    # Structure: {
    #   "net_imbalance": {"passed": bool, "details": {...}},
    #   "sign_convention": {"passed": bool, "details": {...}},
    #   "range_validation": {"passed": bool, "details": {...}},
    #   "missing_data": {"passed": bool, "details": {...}}
    # }
    
    # Overall status
    passed = Column(Boolean, nullable=False)
    error_messages = Column(Text)
