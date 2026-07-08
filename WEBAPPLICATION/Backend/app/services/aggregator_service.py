from datetime import date, datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import numpy as np

from app.db.models.schedule import DailyDispatchSchedule, HourlyDemand, HourlySupply
from app.services.dispatch_forecast_service import DispatchForecastService
from app.schemas.aggregated_schedule import AggregatedScheduleResponse, HourlyAggregation
from app.core.logging import get_logger

logger = get_logger(__name__)

DEMAND_ENTITIES = ["ECG", "NEDCo", "VALCO", "Mines", "Export"]


class AggregatorService:
    def __init__(self):
        self.forecast_service = DispatchForecastService()

    async def aggregate(
        self, db: AsyncSession, schedule_id: int, use_forecast: bool = True
    ) -> Optional[AggregatedScheduleResponse]:
        schedule_obj = await db.get(DailyDispatchSchedule, schedule_id)
        if not schedule_obj:
            return None

        demand_result = await db.execute(
            select(HourlyDemand).where(HourlyDemand.schedule_id == schedule_id)
        )
        demand_rows = demand_result.scalars().all()

        supply_result = await db.execute(
            select(HourlySupply).where(HourlySupply.schedule_id == schedule_id)
        )
        supply_rows = supply_result.scalars().all()

        demand_by_hour: dict[int, dict[str, float]] = {}
        for d in demand_rows:
            if d.hour not in demand_by_hour:
                demand_by_hour[d.hour] = {}
            demand_by_hour[d.hour][d.entity_name] = d.demand_mw

        supply_by_hour: dict[int, float] = {}
        for s in supply_rows:
            if s.hour not in supply_by_hour:
                supply_by_hour[s.hour] = 0.0
            supply_by_hour[s.hour] += s.supply_mw

        forecast_mw = None
        if use_forecast and self.forecast_service.engine and self.forecast_service.engine.is_fitted:
            try:
                result = await self.forecast_service.forecast_for_date(schedule_obj.date)
                if "forecast_mw" in result and len(result["forecast_mw"]) == 24:
                    forecast_mw = result["forecast_mw"]
            except Exception as e:
                logger.warning(f"Forecast fetch failed for aggregation: {e}")

        using_forecast = forecast_mw is not None

        hourly: list[HourlyAggregation] = []
        all_demand_values: list[float] = []
        all_supply_values: list[float] = []

        for hour in range(1, 25):
            entities = demand_by_hour.get(hour, {})

            ecg_val = forecast_mw[hour - 1] if using_forecast and forecast_mw else entities.get("ECG", 0.0)
            nedco_val = entities.get("NEDCo", 0.0)
            valco_val = entities.get("VALCO", 0.0)
            mines_val = entities.get("Mines", 0.0)
            export_val = entities.get("Export", 0.0)

            total_demand = ecg_val + nedco_val + valco_val + mines_val + export_val
            total_supply = supply_by_hour.get(hour, 0.0)
            reserve = round(total_supply - total_demand, 2)
            reserve_pct = round((reserve / total_demand * 100) if total_demand > 0 else 0.0, 2)

            all_demand_values.append(total_demand)
            all_supply_values.append(total_supply)

            hourly.append(HourlyAggregation(
                hour=hour,
                ecg_forecast_mw=round(ecg_val, 2),
                nedco_mw=round(nedco_val, 2),
                valco_mw=round(valco_val, 2),
                mines_mw=round(mines_val, 2),
                export_mw=round(export_val, 2),
                total_demand_mw=round(total_demand, 2),
                total_supply_mw=round(total_supply, 2),
                reserve_mw=round(reserve, 2),
                reserve_pct=reserve_pct,
            ))

        total_energy_demand = round(sum(all_demand_values), 2)
        total_energy_supply = round(sum(all_supply_values), 2)
        avg_demand = round(float(np.mean(all_demand_values)), 2) if all_demand_values else 0.0
        avg_supply = round(float(np.mean(all_supply_values)), 2) if all_supply_values else 0.0

        peak_demand = max(all_demand_values) if all_demand_values else 0.0
        peak_demand_hour = next(
            (h for h in range(1, 25) if abs(hourly[h - 1].total_demand_mw - peak_demand) < 0.01),
            0,
        )

        all_reserves = [h.reserve_mw for h in hourly]
        min_reserve = min(all_reserves) if all_reserves else 0.0
        min_reserve_hour = next(
            (h for h in range(1, 25) if abs(hourly[h - 1].reserve_mw - min_reserve) < 0.01),
            0,
        )

        return AggregatedScheduleResponse(
            schedule_id=schedule_obj.id,
            schedule_date=schedule_obj.date,
            status=schedule_obj.status,
            source_filename=schedule_obj.source_filename,
            operator_notes=schedule_obj.operator_notes,
            hourly=hourly,
            peak_demand_mw=round(peak_demand, 2),
            peak_demand_hour=peak_demand_hour,
            total_energy_demand_mwh=total_energy_demand,
            total_energy_supply_mwh=total_energy_supply,
            avg_demand_mw=avg_demand,
            avg_supply_mw=avg_supply,
            min_reserve_mw=round(min_reserve, 2),
            min_reserve_hour=min_reserve_hour,
            using_forecast=using_forecast,
            computed_at=datetime.utcnow(),
        )
