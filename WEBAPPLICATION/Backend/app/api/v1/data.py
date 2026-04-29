from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import io
from datetime import datetime

from app.api.deps import get_database
from app.schemas.data import (
    DataUploadResponse,
    ValidationReportResponse,
    ValidationSummary
)
from app.services.data_validation import DataValidator
from app.db.models.data import RawDataUpload, ValidationReport, ValidatedData
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/upload", response_model=DataUploadResponse)
async def upload_data(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_database)
):
    """
    Upload and validate SCADA CSV data.
    
    - **file**: CSV file with time-series load data
    
    Returns upload status and validation summary.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Parse CSV
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
        
        # Create upload record
        upload = RawDataUpload(
            filename=file.filename,
            file_size_bytes=file_size,
            row_count=len(df),
            status="validating"
        )
        
        db.add(upload)
        await db.commit()
        await db.refresh(upload)
        
        # Validate data
        validator = DataValidator()
        validation_results = validator.validate_csv(df)
        
        # Create validation report
        report = ValidationReport(
            upload_id=upload.id,
            total_rows=validation_results["total_rows"],
            valid_rows=validation_results["valid_rows"],
            invalid_rows=validation_results["invalid_rows"],
            anomaly_count=validation_results.get("anomaly_count", 0),
            validation_summary=validation_results["validation_checks"],
            passed=validation_results["passed"],
            error_messages="; ".join(validation_results.get("error_messages", []))
        )
        
        db.add(report)
        await db.commit()
        await db.refresh(report)
        
        # Update upload status
        upload.status = "validated" if validation_results["passed"] else "failed"
        upload.validation_report_id = report.id
        await db.commit()
        
        # Store validated data if passed
        if validation_results["passed"]:
            await _store_validated_data(df, upload.id, db)
        
        return DataUploadResponse(
            upload_id=upload.id,
            filename=file.filename,
            file_size_bytes=file_size,
            row_count=len(df),
            status=upload.status,
            message=f"Validation {'passed' if validation_results['passed'] else 'failed'}. Report ID: {report.id}",
            health_grade=validation_results.get("health_grade", "F"),
            impact_summary=validation_results.get("impact_summary")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/validation/{upload_id}", response_model=ValidationReportResponse)
async def get_validation_report(
    upload_id: int,
    db: AsyncSession = Depends(get_database)
):
    """
    Get validation report for an uploaded file.
    
    - **upload_id**: ID of the uploaded file
    """
    from sqlalchemy import select
    
    # Get validation report
    result = await db.execute(
        select(ValidationReport).where(ValidationReport.upload_id == upload_id)
    )
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(status_code=404, detail="Validation report not found")
    
    # Convert validation checks to list of ValidationSummary
    validation_checks = [
        ValidationSummary(
            check_name=name,
            passed=check.get("passed", True),
            details=check.get("details", {})
        )
        for name, check in report.validation_summary.items()
    ]
    
    return ValidationReportResponse(
        report_id=report.id,
        upload_id=report.upload_id,
        created_at=report.created_at,
        total_rows=report.total_rows,
        valid_rows=report.valid_rows,
        invalid_rows=report.invalid_rows,
        anomaly_count=report.anomaly_count,
        passed=report.passed,
        validation_checks=validation_checks,
        error_messages=report.error_messages
    )


@router.get("/uploads")
async def list_uploads(
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_database)
):
    """
    List all uploaded files.
    
    - **skip**: Number of records to skip
    - **limit**: Maximum number of records to return
    """
    from sqlalchemy import select
    
    result = await db.execute(
        select(RawDataUpload)
        .order_by(RawDataUpload.upload_timestamp.desc())
        .offset(skip)
        .limit(limit)
    )
    uploads = result.scalars().all()
    
    return {
        "uploads": [
            {
                "id": u.id,
                "filename": u.filename,
                "upload_timestamp": u.upload_timestamp,
                "row_count": u.row_count,
                "status": u.status
            }
            for u in uploads
        ]
    }


@router.get("/latest")
async def get_latest_data(
    limit: int = 100,
    db: AsyncSession = Depends(get_database)
):
    """
    Get latest validated data points.
    
    - **limit**: Maximum number of records to return
    """
    from sqlalchemy import select
    
    result = await db.execute(
        select(ValidatedData)
        .order_by(ValidatedData.timestamp.desc())
        .limit(limit)
    )
    data = result.scalars().all()
    
    return [
        {
            "timestamp": d.timestamp,
            "total_load_mw": d.total_load_mw,
            "line1_mw": d.line1_mw,
            "line2_mw": d.line2_mw,
            "line3_mw": d.line3_mw,
            "voltage_kv": d.voltage_kv,
            "current_a": d.current_a,
            "temperature_c": d.temperature_c,
            "frequency_hz": d.frequency_hz,
            "is_anomaly": d.is_anomaly
        }
        for d in data
    ]


@router.post("/reset")
async def reset_system_data(
    db: AsyncSession = Depends(get_database)
):
    """
    Factory Reset: Wipes all uploaded data and restores the original Community Load baseline.
    """
    import subprocess
    import os
    from app.services.forecast_service import ForecastService
    
    try:
        # Path to the restoration script
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts", "restore_community_data.py")
        
        # Execute restoration script
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Restoration script failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Reset failed: {result.stderr}")
        
        # Clear metrics cache
        fs = ForecastService()
        fs._cache["metrics"]["data"] = None
            
        return {"status": "success", "message": "System reverted to Community Load baseline"}
        
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System reset failed: {str(e)}")


async def _store_validated_data(df: pd.DataFrame, upload_id: int, db: AsyncSession):
    """Store validated data in the database."""
    try:
        # Standardize columns for mapping
        df = df.copy()
        df.columns = [col.upper() for col in df.columns]
        
        # Prepare data for insertion
        records = []
        
        for idx, row in df.iterrows():
            record = ValidatedData(
                upload_id=upload_id,
                timestamp=pd.to_datetime(row.get("TIMESTAMP")),
                total_load_mw=row.get("TOTAL_LOAD_MW"),
                line1_mw=row.get("LINE1_MW"),
                line2_mw=row.get("LINE2_MW"),
                line3_mw=row.get("LINE3_MW"),
                voltage_kv=row.get("VOLTAGE_KV"),
                current_a=row.get("CURRENT_A"),
                temperature_c=row.get("TEMP_C") if row.get("TEMP_C") is not None else row.get("TEMPERATURE_C"),
                frequency_hz=row.get("FREQ_HZ") if row.get("FREQ_HZ") is not None else row.get("FREQUENCY_HZ"),
                is_anomaly=False,
                validation_flags={}
            )
            records.append(record)
        
        # Bulk insert
        db.add_all(records)
        await db.commit()
        
        logger.info(f"Stored {len(records)} validated records for upload {upload_id}")
        
    except Exception as e:
        logger.error(f"Error storing validated data: {str(e)}")
        await db.rollback()
        raise
