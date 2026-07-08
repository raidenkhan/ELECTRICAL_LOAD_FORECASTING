import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.api.deps import get_database
from app.schemas.schedule import (
    ScheduleUploadResponse, ScheduleDetail, CellUpdateRequest, ConfirmRequest,
    ReviseRequest,
)
from app.services.schedule_service import ScheduleService
from app.services.dispatch_forecast_service import DispatchForecastService
from app.services.aggregator_service import AggregatorService
from app.services.baseload_service import BaseloadService
from app.services.audit_service import AuditService
from app.schemas.aggregated_schedule import AggregatedScheduleResponse
from app.schemas.audit_log import AuditLogEntry, ChainVerificationResult
from app.db.models.schedule import DailyDispatchSchedule, HourlyDemand, HourlySupply
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)
schedule_service = ScheduleService()
forecast_service = DispatchForecastService()
aggregator_service = AggregatorService()
baseload_service = BaseloadService()
audit_service = AuditService()


@router.post("/upload", response_model=ScheduleUploadResponse)
async def upload_schedule(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_database),
):
    """
    Upload a Daily Dispatch Schedule Excel file.

    Accepts `.xlsx` files in the ECG Daily Demand Data Sheet format.
    Parses the 24-hour demand and supply rows and stores them in the database.
    """
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are supported")

    try:
        content = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        schedule_obj = await schedule_service.parse_and_store(
            db=db,
            filepath=tmp_path,
            filename=file.filename,
        )

        await audit_service.log(
            db, schedule_obj.id, "upload",
            f"Schedule uploaded: {file.filename}",
            {"filename": file.filename, "date": str(schedule_obj.date)},
        )
        await db.commit()

        return ScheduleUploadResponse(
            id=schedule_obj.id,
            date=schedule_obj.date,
            status=schedule_obj.status,
            source_filename=schedule_obj.source_filename,
            demand_count=0,
            supply_count=0,
            message=f"Schedule for {schedule_obj.date.isoformat()} stored successfully (id={schedule_obj.id})",
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Schedule upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if "tmp_path" in locals():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@router.patch("/{schedule_id}/cell", response_model=ScheduleDetail)
async def update_cell(
    schedule_id: int,
    body: CellUpdateRequest,
    db: AsyncSession = Depends(get_database),
):
    schedule_obj = await db.get(DailyDispatchSchedule, schedule_id)
    if not schedule_obj:
        raise HTTPException(status_code=404, detail="Schedule not found")
    if schedule_obj.status == "confirmed":
        raise HTTPException(status_code=400, detail="Cannot edit a confirmed schedule")

    old_value = None
    if body.table == "demand":
        result = await db.execute(
            select(HourlyDemand).where(
                HourlyDemand.schedule_id == schedule_id,
                HourlyDemand.hour == body.hour,
                HourlyDemand.entity_name == body.entity_name,
            )
        )
        row = result.scalar_one_or_none()
        if not row:
            raise HTTPException(status_code=404, detail="Demand cell not found")
        old_value = row.demand_mw
        row.demand_mw = body.value
    elif body.table == "supply":
        result = await db.execute(
            select(HourlySupply).where(
                HourlySupply.schedule_id == schedule_id,
                HourlySupply.hour == body.hour,
                HourlySupply.plant_name == body.entity_name,
            )
        )
        row = result.scalar_one_or_none()
        if not row:
            raise HTTPException(status_code=404, detail="Supply cell not found")
        old_value = row.supply_mw
        row.supply_mw = body.value
    else:
        raise HTTPException(status_code=400, detail="table must be 'demand' or 'supply'")

    await db.flush()
    await audit_service.log(
        db, schedule_id, "cell_update",
        f"Updated {body.table} cell {body.entity_name} H{body.hour}: {old_value} \u2192 {body.value} MW",
        {
            "table": body.table,
            "entity_name": body.entity_name,
            "hour": body.hour,
            "old_value": old_value,
            "new_value": body.value,
        },
    )
    await db.commit()
    result_dict = await schedule_service.get_schedule(db, schedule_id)
    return result_dict


@router.post("/{schedule_id}/auto-fill-forecast", response_model=ScheduleDetail)
async def auto_fill_forecast(
    schedule_id: int,
    db: AsyncSession = Depends(get_database),
):
    """
    Auto-fill ECG demand cells with forecast values from the DecomEngine.

    Calls the dispatch forecast engine for the schedule's date and updates
    all 24 ECG hourly demand rows with predicted values, marking them as
    `is_forecasted = True`. Only works on draft schedules.
    """
    schedule_obj = await db.get(DailyDispatchSchedule, schedule_id)
    if not schedule_obj:
        raise HTTPException(status_code=404, detail="Schedule not found")
    if schedule_obj.status == "confirmed":
        raise HTTPException(status_code=400, detail="Cannot modify a confirmed schedule")

    if not forecast_service.engine or not forecast_service.engine.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Forecast engine not trained. Run train_ecg_engine.py first.",
        )

    result = await forecast_service.forecast_for_date(schedule_obj.date)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    forecast_mw = result["forecast_mw"]
    if len(forecast_mw) != 24:
        raise HTTPException(status_code=500, detail="Forecast returned invalid length")

    # Update ECG demand rows with forecast values
    for hour in range(1, 25):
        stmt = select(HourlyDemand).where(
            HourlyDemand.schedule_id == schedule_id,
            HourlyDemand.hour == hour,
            HourlyDemand.entity_name == "ECG",
        )
        ecg_row = (await db.execute(stmt)).scalar_one_or_none()
        if ecg_row:
            ecg_row.demand_mw = round(forecast_mw[hour - 1], 2)
            ecg_row.is_forecasted = True

    await db.flush()
    await audit_service.log(
        db, schedule_id, "forecast_fill",
        "ECG demand auto-filled from DecomEngine forecast",
    )
    await db.commit()
    result_dict = await schedule_service.get_schedule(db, schedule_id)
    return result_dict


@router.post("/{schedule_id}/confirm", response_model=ScheduleDetail)
async def confirm_schedule(
    schedule_id: int,
    body: ConfirmRequest = ConfirmRequest(),
    db: AsyncSession = Depends(get_database),
):
    """
    Confirm a dispatch schedule. Once confirmed, cells cannot be edited.

    - **operator_notes**: optional reason or note for the confirmation
    """
    schedule_obj = await db.get(DailyDispatchSchedule, schedule_id)
    if not schedule_obj:
        raise HTTPException(status_code=404, detail="Schedule not found")
    if schedule_obj.status == "confirmed":
        raise HTTPException(status_code=400, detail="Schedule is already confirmed")

    schedule_obj.status = "confirmed"
    if body.operator_notes:
        schedule_obj.operator_notes = body.operator_notes
    await db.flush()
    await audit_service.log(
        db, schedule_id, "confirm",
        f"Schedule confirmed",
        {"operator_notes": body.operator_notes},
    )
    await db.commit()

    result_dict = await schedule_service.get_schedule(db, schedule_id)
    return result_dict


@router.post("/{schedule_id}/revise", response_model=ScheduleDetail)
async def revise_schedule(
    schedule_id: int,
    body: ReviseRequest,
    db: AsyncSession = Depends(get_database),
):
    """
    Revise a confirmed dispatch schedule. Sets status back to 'draft' so cells
    can be edited again, and records the revision reason.

    - **operator_notes**: required reason for the revision
    """
    schedule_obj = await db.get(DailyDispatchSchedule, schedule_id)
    if not schedule_obj:
        raise HTTPException(status_code=404, detail="Schedule not found")
    if schedule_obj.status == "draft":
        raise HTTPException(status_code=400, detail="Schedule is already in draft state")

    schedule_obj.status = "draft"
    schedule_obj.operator_notes = body.operator_notes
    await db.flush()
    await audit_service.log(
        db, schedule_id, "revise",
        f"Schedule revised: {body.operator_notes}",
        {"reason": body.operator_notes},
    )
    await db.commit()

    result_dict = await schedule_service.get_schedule(db, schedule_id)
    return result_dict


@router.get("/latest", response_model=ScheduleDetail)
async def get_latest_schedule(
    db: AsyncSession = Depends(get_database),
):
    """
    Get the most recently uploaded dispatch schedule.
    """
    result = await db.execute(
        select(DailyDispatchSchedule)
        .order_by(desc(DailyDispatchSchedule.created_at))
        .limit(1)
    )
    schedule_obj = result.scalar_one_or_none()
    if not schedule_obj:
        raise HTTPException(status_code=404, detail="No schedules found")

    await schedule_service._merge_baseload_plants(db, schedule_obj.id)
    await db.commit()
    result_dict = await schedule_service.get_schedule(db, schedule_obj.id)
    return result_dict


@router.get("/{schedule_id}/aggregated", response_model=AggregatedScheduleResponse)
async def get_aggregated_schedule(
    schedule_id: int,
    db: AsyncSession = Depends(get_database),
    use_forecast: bool = True,
):
    """
    Get the aggregated dispatch schedule with computed totals, reserves, and
    forecast integration.

    - **use_forecast** (query param, default `true`): if true, replaces the
      uploaded ECG values with the DecomEngine forecast for the schedule's date.
      Set to `false` to use the uploaded ECG values directly.

    Returns per-hour breakdowns:
    - **total_demand_mw** = forecasted ECG + NEDCo + VALCO + Mines + Export
    - **total_supply_mw** = sum of all embedded sources per hour
    - **reserve_mw** = total_supply - total_demand
    """
    result = await aggregator_service.aggregate(db, schedule_id, use_forecast=use_forecast)
    if result is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return result


@router.get("/{schedule_id}", response_model=ScheduleDetail)
async def get_schedule(
    schedule_id: int,
    db: AsyncSession = Depends(get_database),
):
    """
    Get a parsed dispatch schedule by ID.

    Returns the full schedule with all hourly demand and supply rows.
    Baseload plants are auto-merged if not yet present.
    """
    await schedule_service._merge_baseload_plants(db, schedule_id)
    await db.commit()
    result = await schedule_service.get_schedule(db, schedule_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return result


@router.get("/{schedule_id}/audit-logs", response_model=list[AuditLogEntry])
async def get_audit_logs(
    schedule_id: int,
    db: AsyncSession = Depends(get_database),
):
    """Get the audit trail for a dispatch schedule."""
    logs = await audit_service.get_for_schedule(db, schedule_id)
    return logs


@router.get("/{schedule_id}/audit-logs/verify", response_model=ChainVerificationResult)
async def verify_audit_chain(
    schedule_id: int,
    db: AsyncSession = Depends(get_database),
):
    """Verify the hash chain integrity for a schedule's audit trail."""
    result = await audit_service.verify_chain(db, schedule_id)
    return result


@router.get("", response_model=list[ScheduleDetail])
async def list_schedules(
    db: AsyncSession = Depends(get_database),
):
    """List all dispatch schedules, newest first."""
    result = await db.execute(
        select(DailyDispatchSchedule)
        .order_by(desc(DailyDispatchSchedule.created_at))
    )
    schedules = result.scalars().all()
    result_list = []
    for s in schedules:
        await schedule_service._merge_baseload_plants(db, s.id)
        detail = await schedule_service.get_schedule(db, s.id)
        if detail:
            result_list.append(detail)
    await db.commit()
    return result_list
