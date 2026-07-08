"""Rolling forecast metrics with drift detection."""
from datetime import date, timedelta, datetime
from typing import Dict, Any, List, Optional
import numpy as np

from app.db.models.forecast_metrics import ForecastMetrics
from app.db.session import AsyncSessionLocal
from sqlalchemy import select, delete, desc, func
from app.core.logging import get_logger

logger = get_logger(__name__)

BASELINE_MAE_MW = 67.0  # DLinear+H10 D+1 MAE from stress tests
DRIFT_THRESHOLD = 0.10   # 10% degradation triggers alert


class MetricsService:
    async def record(self, forecast_date: date, horizon: str,
                     forecast_mw: List[float], actual_mw: Optional[List[float]] = None):
        try:
            fw = np.array(forecast_mw, dtype=np.float32)
            mae = 0.0
            mape = None
            if actual_mw is not None:
                aw = np.array(actual_mw, dtype=np.float32)
                if len(aw) == len(fw) and len(aw) > 0:
                    err = np.abs(aw - fw)
                    mae = float(np.mean(err))
                    mask = aw > 0
                    if mask.any():
                        mape = float(np.mean(err[mask] / aw[mask]) * 100)

            async with AsyncSessionLocal() as db:
                entry = ForecastMetrics(
                    forecast_date=forecast_date,
                    horizon=horizon,
                    actual_mw=actual_mw,
                    forecast_mw=forecast_mw,
                    mae=mae,
                    mape=mape,
                    engine="dlinear_h10",
                    created_at=datetime.utcnow(),
                )
                db.add(entry)
                await db.commit()
                logger.info(f"Metrics recorded: {forecast_date} {horizon} MAE={mae:.1f} MAPE={mape}")
        except Exception as e:
            logger.warning(f"Metrics record failed: {e}")

    async def get_rolling(self, horizon: str = "24h",
                          window_days: int = 30) -> Dict[str, Any]:
        try:
            cutoff = date.today() - timedelta(days=window_days)
            async with AsyncSessionLocal() as db:
                stmt = (
                    select(ForecastMetrics)
                    .where(ForecastMetrics.horizon == horizon,
                           ForecastMetrics.forecast_date >= cutoff,
                           ForecastMetrics.actual_mw.isnot(None))
                    .order_by(ForecastMetrics.forecast_date)
                )
                result = await db.execute(stmt)
                rows = result.scalars().all()
                if not rows:
                    return {"count": 0, "mae_avg": None, "mape_avg": None, "mae_list": [], "mape_list": []}
                maes = [r.mae for r in rows if r.mae is not None]
                mapes = [r.mape for r in rows if r.mape is not None]
                return {
                    "count": len(rows),
                    "window_days": window_days,
                    "mae_avg": float(np.mean(maes)) if maes else None,
                    "mae_std": float(np.std(maes)) if len(maes) > 1 else None,
                    "mape_avg": float(np.mean(mapes)) if mapes else None,
                    "mae_list": [{"date": str(r.forecast_date), "mae": r.mae, "mape": r.mape} for r in rows if r.mae is not None],
                }
        except Exception as e:
            logger.warning(f"Rolling metrics query failed: {e}")
            return {"count": 0, "error": str(e)}

    async def check_drift(self, baseline: float = BASELINE_MAE_MW,
                          threshold: float = DRIFT_THRESHOLD) -> Dict[str, Any]:
        recent = await self.get_rolling("24h", window_days=30)
        current = recent.get("mae_avg")
        if current is None:
            return {"drift_detected": False, "reason": "insufficient_data", "current_mae": None, "baseline": baseline}
        degradation = (current - baseline) / baseline
        drifted = degradation > threshold
        return {
            "drift_detected": drifted,
            "current_mae": round(current, 1),
            "baseline_mae": baseline,
            "degradation_pct": round(degradation * 100, 1),
            "threshold_pct": threshold * 100,
            "count": recent.get("count", 0),
            "window_days": 30,
        }

    async def latest(self, horizon: str = "24h") -> Optional[Dict[str, Any]]:
        try:
            async with AsyncSessionLocal() as db:
                stmt = (
                    select(ForecastMetrics)
                    .where(ForecastMetrics.horizon == horizon,
                           ForecastMetrics.actual_mw.isnot(None))
                    .order_by(desc(ForecastMetrics.created_at))
                    .limit(1)
                )
                row = (await db.execute(stmt)).scalar_one_or_none()
                if row:
                    return {"date": str(row.forecast_date), "mae": row.mae, "mape": row.mape, "horizon": row.horizon}
                return None
        except Exception as e:
            logger.warning(f"Latest metrics query failed: {e}")
            return None
