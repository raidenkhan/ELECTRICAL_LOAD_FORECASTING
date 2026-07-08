from fastapi import APIRouter, Query
from typing import Optional
from datetime import date, timedelta
import numpy as np

from app.ml.metrics_service import MetricsService
from app.db.session import AsyncSessionLocal
from app.db.models.forecast_metrics import ForecastMetrics
from sqlalchemy import select, func

router = APIRouter()
metrics_service = MetricsService()

HOUR_COUNT = 24

# Hardcoded per-fold historical metrics (static — training benchmarks)
FOLD_METRICS = {
    "24h": [
        {"fold": "F1", "period": "2020 H1", "mae": 72, "mape": 4.2, "context": "COVID low demand"},
        {"fold": "F2", "period": "2021 H1", "mae": 64, "mape": 3.1, "context": "Stable recovery"},
        {"fold": "F3", "period": "2022 H1", "mae": 71, "mape": 3.6, "context": "Moderate growth"},
        {"fold": "F4", "period": "2023 H1", "mae": 68, "mape": 3.3, "context": "Steady growth"},
        {"fold": "F5", "period": "2024 H1", "mae": 82, "mape": 4.0, "context": "High growth"},
        {"fold": "F6", "period": "2025 H1", "mae": 88, "mape": 4.1, "context": "Growth to 2026"},
    ],
    "7d": [
        {"fold": "F1", "period": "2020 H1", "mae": 290, "mape": 8.8, "context": "Low base + COVID"},
        {"fold": "F2", "period": "2021 H1", "mae": 210, "mape": 6.2, "context": "Stable recovery"},
        {"fold": "F3", "period": "2022 H1", "mae": 230, "mape": 7.0, "context": "Moderate growth"},
        {"fold": "F4", "period": "2023 H1", "mae": 220, "mape": 6.6, "context": "Steady growth"},
        {"fold": "F5", "period": "2024 H1", "mae": 260, "mape": 7.8, "context": "High growth"},
        {"fold": "F6", "period": "2025 H1", "mae": 310, "mape": 8.5, "context": "Growth discontinuity"},
    ],
    "30d": [
        {"fold": "F1", "period": "2020 H1", "mae": 350, "mape": 11.2, "context": "Low base"},
        {"fold": "F2", "period": "2021 H1", "mae": 260, "mape": 8.0, "context": "Stable"},
        {"fold": "F3", "period": "2022 H1", "mae": 280, "mape": 8.8, "context": "Moderate growth"},
        {"fold": "F4", "period": "2023 H1", "mae": 270, "mape": 8.4, "context": "Steady growth"},
        {"fold": "F5", "period": "2024 H1", "mae": 310, "mape": 9.5, "context": "High growth"},
        {"fold": "F6", "period": "2025 H1", "mae": 380, "mape": 10.2, "context": "Growth discontinuity"},
    ],
    "90d": [
        {"fold": "F1", "period": "2020 H1", "mae": 480, "mape": 15.0, "context": "Low base"},
        {"fold": "F2", "period": "2021 H1", "mae": 360, "mape": 11.0, "context": "Stable"},
        {"fold": "F3", "period": "2022 H1", "mae": 390, "mape": 12.0, "context": "Moderate growth"},
        {"fold": "F4", "period": "2023 H1", "mae": 380, "mape": 11.6, "context": "Steady growth"},
        {"fold": "F5", "period": "2024 H1", "mae": 430, "mape": 13.0, "context": "High growth"},
        {"fold": "F6", "period": "2025 H1", "mae": 520, "mape": 14.5, "context": "Growth discontinuity"},
    ],
}

# Nominal expected MAE per horizon (from training benchmarks)
NOMINAL_MAE = {"24h": 67, "7d": 251, "30d": 302, "90d": 420}
NOMINAL_MAPE = {"24h": 2.8, "7d": 7.6, "30d": 9.2, "90d": 12.8}


@router.get("/metrics/rolling")
async def get_rolling_metrics(
    horizon: str = Query("24h", description="Forecast horizon"),
    window_days: int = Query(30, description="Rolling window in days"),
):
    return await metrics_service.get_rolling(horizon=horizon, window_days=window_days)


@router.get("/metrics/drift")
async def check_drift():
    return await metrics_service.check_drift()


@router.get("/metrics/latest")
async def get_latest_metrics(horizon: str = Query("24h")):
    return await metrics_service.latest(horizon=horizon)


@router.post("/metrics/record")
async def record_metrics(
    forecast_date: str,
    horizon: str = "24h",
    forecast_mw: list[float] = [],
    actual_mw: Optional[list[float]] = None,
):
    dt = date.fromisoformat(forecast_date)
    await metrics_service.record(dt, horizon, forecast_mw, actual_mw)
    return {"status": "recorded"}


@router.get("/metrics/by-hour")
async def get_metrics_by_hour(
    horizon: str = Query("24h", description="Forecast horizon"),
    window_days: int = Query(90, description="Lookback window"),
):
    """Compute per-hour MAE from forecast_metrics table."""
    try:
        cutoff = date.today() - timedelta(days=window_days)
        async with AsyncSessionLocal() as db:
            stmt = (
                select(ForecastMetrics)
                .where(
                    ForecastMetrics.horizon == horizon,
                    ForecastMetrics.forecast_date >= cutoff,
                    ForecastMetrics.actual_mw.isnot(None),
                )
                .order_by(ForecastMetrics.forecast_date.desc())
            )
            rows = (await db.execute(stmt)).scalars().all()

        if not rows:
            return {"by_hour": [], "source": "nominal"}

        hour_sums = np.zeros(HOUR_COUNT, dtype=np.float64)
        hour_counts = np.zeros(HOUR_COUNT, dtype=np.int32)

        for r in rows:
            if r.actual_mw is None or r.forecast_mw is None:
                continue
            fw = np.array(r.forecast_mw, dtype=np.float64)
            aw = np.array(r.actual_mw, dtype=np.float64)
            if len(fw) != HOUR_COUNT or len(aw) != HOUR_COUNT:
                continue
            err = np.abs(fw - aw)
            valid = np.isfinite(err)
            hour_sums[valid] += err[valid]
            hour_counts[valid] += 1

        by_hour = []
        for h in range(HOUR_COUNT):
            mae = round(float(hour_sums[h] / hour_counts[h]), 1) if hour_counts[h] > 0 else None
            by_hour.append({"hour": h + 1, "mae": mae, "n": int(hour_counts[h])})

        return {"by_hour": by_hour, "source": "db", "n_rows": len(rows)}
    except Exception as e:
        return {"by_hour": [], "source": "error", "error": str(e)}


@router.get("/metrics/by-dow")
async def get_metrics_by_dow(
    horizon: str = Query("24h", description="Forecast horizon"),
    window_days: int = Query(90, description="Lookback window"),
):
    """Compute per-day-of-week MAE from forecast_metrics table."""
    try:
        cutoff = date.today() - timedelta(days=window_days)
        async with AsyncSessionLocal() as db:
            stmt = (
                select(ForecastMetrics)
                .where(
                    ForecastMetrics.horizon == horizon,
                    ForecastMetrics.forecast_date >= cutoff,
                    ForecastMetrics.actual_mw.isnot(None),
                )
                .order_by(ForecastMetrics.forecast_date.desc())
            )
            rows = (await db.execute(stmt)).scalars().all()

        if not rows:
            return {"by_dow": [], "source": "nominal"}

        dow_sums = {}
        dow_counts = {}
        DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for r in rows:
            dow = r.forecast_date.weekday()
            if dow not in dow_sums:
                dow_sums[dow] = 0.0
                dow_counts[dow] = 0
            dow_sums[dow] += r.mae
            dow_counts[dow] += 1

        by_dow = []
        for d in range(7):
            mae = round(dow_sums[d] / dow_counts[d], 1) if dow_counts.get(d, 0) > 0 else None
            by_dow.append({"day": DOW_NAMES[d], "mae": mae, "n": dow_counts.get(d, 0)})

        return {"by_dow": by_dow, "source": "db", "n_rows": len(rows)}
    except Exception as e:
        return {"by_dow": [], "source": "error", "error": str(e)}


@router.get("/metrics/overview")
async def get_metrics_overview(horizon: str = Query("24h")):
    """Combined overview: nominal MAE/MAPE, folds, and live rolling if available."""
    rolling = await metrics_service.get_rolling(horizon=horizon, window_days=30)
    drift = await metrics_service.check_drift()
    return {
        "horizon": horizon,
        "nominal_mae": NOMINAL_MAE.get(horizon),
        "nominal_mape": NOMINAL_MAPE.get(horizon),
        "folds": FOLD_METRICS.get(horizon, []),
        "rolling": rolling,
        "drift": drift,
    }
