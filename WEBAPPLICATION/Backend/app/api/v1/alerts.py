from fastapi import APIRouter
from typing import List, Dict, Any
from datetime import datetime, date, timedelta

from app.services.dispatch_forecast_service import DispatchForecastService
from app.services.baseload_service import BaseloadService
from app.db.session import AsyncSessionLocal
from sqlalchemy import select, func
from app.db.models.baseload import BaseloadPlant
from app.db.models.ecg_history import EcgHistoricalDemand

router = APIRouter()
forecast_service = DispatchForecastService()
baseload_service = BaseloadService()

STALE_DATA_HOURS = 48
FALLBACK_TRIGGER_MAE = 150


@router.get("/")
async def get_active_alerts():
    alerts = []

    try:
        # 1. Data freshness: check most recent ECG history
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(func.max(EcgHistoricalDemand.date)).where(
                    EcgHistoricalDemand.demand_mw.isnot(None)
                )
                max_date = (await db.execute(stmt)).scalar_one_or_none()
                if max_date is not None:
                    last_dt = datetime.combine(max_date, datetime.min.time())
                    hours_since = (datetime.utcnow() - last_dt).total_seconds() / 3600
                    if hours_since > STALE_DATA_HOURS:
                        severity = "critical" if hours_since > 72 else "warning"
                        alerts.append({
                            "id": 1,
                            "type": severity,
                            "title": f"Data Stale — no update in {int(hours_since)}h",
                            "detail": f"Last ECG demand record: {max_date.isoformat()}. Pipeline may be down.",
                            "time": f"{int(hours_since)}h ago",
                            "actions": [{"label": "Check Pipeline", "primary": True}],
                        })
        except Exception as e:
            pass

        # 2. Capacity margin check
        result = await forecast_service.forecast_tomorrow()
        forecast_mw = result.get("forecast_mw", [])

        if forecast_mw:
            peak_val = max(forecast_mw)
            peak_idx = int(forecast_mw.index(peak_val))

            total_capacity = 4000.0
            try:
                async with AsyncSessionLocal() as db:
                    stmt = select(BaseloadPlant).where(BaseloadPlant.is_active == True)
                    rows = (await db.execute(stmt)).scalars().all()
                    if rows:
                        total_capacity = sum(r.capacity_mw for r in rows)
            except Exception:
                pass

            margin_pct = (total_capacity - peak_val) / total_capacity * 100
            if margin_pct < 7.0:
                severity = "critical" if margin_pct < 3.0 else "warning"
                alerts.append({
                    "id": 2,
                    "type": severity,
                    "title": f"{severity.upper()}: Capacity Margin Breach",
                    "detail": (f"Peak: {int(peak_val):,} MW at H{peak_idx:02d} | "
                               f"Capacity: {int(total_capacity):,} MW | Margin: {margin_pct:.1f}%"),
                    "time": "Now",
                    "actions": [
                        {"label": "Scale Generation", "primary": True},
                        {"label": "Acknowledge"},
                    ],
                })

        # 3. Engine health
        engine = forecast_service.engine
        if engine and engine.is_fitted:
            health = engine.health()
            mae = health.get("mae_24h")

            # MAE drift alert
            if mae and mae > FALLBACK_TRIGGER_MAE:
                alerts.append({
                    "id": 3,
                    "type": "warning",
                    "title": f"DLinear MAE elevated: {mae:.0f} MW",
                    "detail": (f"Rolling 24h MAE: {mae:.0f} MW (threshold: {FALLBACK_TRIGGER_MAE} MW). "
                               f"Inferences: {health.get('inference_count', '?')}. "
                               f"Corrector: {'trained' if health.get('is_trained') else 'untrained'}"),
                    "time": "Now",
                    "actions": [{"label": "View Metrics"}],
                })

            # Corrector staleness
            if health.get("stale", True) and health.get("is_trained", False):
                alerts.append({
                    "id": 5,
                    "type": "warning",
                    "title": "Corrector State Stale",
                    "detail": "ARD corrector has not received feedback in 7+ days.",
                    "time": "Now",
                    "actions": [{"label": "Reset Corrector"}],
                })

            # Checkpoint age
            import os
            from pathlib import Path
            ckpt_dir = Path(engine.checkpoint_dir)
            ckpts = sorted(ckpt_dir.glob("h10_Fold_*.pt")) if ckpt_dir.exists() else []
            if ckpts:
                latest_mtime = max(os.path.getmtime(c) for c in ckpts)
                age_days = (datetime.utcnow() - datetime.fromtimestamp(latest_mtime)).days
                if age_days > 180:
                    alerts.append({
                        "id": 6,
                        "type": "info",
                        "title": f"Checkpoints {age_days}d old — retraining due",
                        "detail": f"Last checkpoint modified {age_days} days ago. Retrain every 180 days.",
                        "time": f"{age_days}d ago",
                        "actions": [{"label": "Retrain Now", "primary": True}],
                    })
        else:
            alerts.append({
                "id": 4,
                "type": "critical",
                "title": "DLinear Engine Offline",
                "detail": "Engine not fitted — forecasts falling back to statistical method.",
                "time": "Now",
                "actions": [{"label": "Restart Service", "primary": True}],
            })

    except Exception as e:
        alerts.append({
            "id": 0,
            "type": "critical",
            "title": "Alert System Error",
            "detail": f"Failed to evaluate alerts: {str(e)}",
            "time": "Now",
            "actions": [{"label": "Restart Service", "primary": True}],
        })

    return alerts
