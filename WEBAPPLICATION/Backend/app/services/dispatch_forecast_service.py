from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import os
from datetime import date, timedelta, datetime, timezone

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.dlinear_engine import DLinearEngine
from app.ml.metrics_service import MetricsService
from app.services.weather_service import WeatherService
from app.db.models.ecg_history import EcgHistoricalDemand
from app.db.models.forecast_cache import ForecastCache
from app.db.session import AsyncSessionLocal
from sqlalchemy import select, delete

logger = get_logger(__name__)

HISTORY_NEEDED = 200  # hours to fetch for DLinear (168 window + 32 buffer)


class DispatchForecastService:
    def __init__(self):
        self.engine: Optional[DLinearEngine] = None
        self.weather_service = WeatherService()
        self.metrics = MetricsService()
        self._load_engine()

    def _load_engine(self):
        self.engine = DLinearEngine()
        if not self.engine.is_fitted:
            logger.warning("DLinearEngine not fitted. Check checkpoints.")

    async def _cache_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(ForecastCache).where(ForecastCache.cache_key == cache_key)
                result = await db.execute(stmt)
                entry = result.scalar_one_or_none()
                if entry:
                    logger.info(f"Cache HIT for {cache_key}")
                    return dict(entry.data)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
        return None

    async def _cache_set(self, cache_key: str, horizon: str, data: Dict[str, Any]):
        try:
            async with AsyncSessionLocal() as db:
                await db.execute(delete(ForecastCache).where(ForecastCache.cache_key == cache_key))
                entry = ForecastCache(
                    cache_key=cache_key,
                    horizon=horizon,
                    forecast_date=date.today(),
                    data=data,
                    created_at=datetime.utcnow(),
                )
                db.add(entry)
                await db.commit()
                logger.info(f"Cached {cache_key} ({len(str(data))} bytes)")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    async def _cache_clear_all(self):
        try:
            async with AsyncSessionLocal() as db:
                await db.execute(delete(ForecastCache))
                await db.commit()
                logger.info("Forecast cache cleared")
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")

    async def _fetch_history(self, min_hours: int = HISTORY_NEEDED) -> Optional[pd.DataFrame]:
        try:
            async with AsyncSessionLocal() as db:
                stmt = (
                    select(EcgHistoricalDemand)
                    .order_by(EcgHistoricalDemand.date.desc(), EcgHistoricalDemand.hour.desc())
                    .limit(min_hours)
                )
                result = await db.execute(stmt)
                rows = result.scalars().all()
                if not rows or len(rows) < 168:
                    logger.warning(f"Only {len(rows) if rows else 0} history rows — need 168+")
                    return None
                rows = list(reversed(rows))
                dates = [pd.Timestamp(r.date, hour=r.hour - 1) for r in rows]
                df = pd.DataFrame({
                    "date": dates,
                    "demand_mw": [float(r.demand_mw) for r in rows],
                    "temperature_c": [float(r.temperature_c) if r.temperature_c is not None else None for r in rows],
                })
                null_mask = df["temperature_c"].isna()
                if null_mask.any():
                    fallback = self.weather_service.seasonal_fallback(dates[0].to_pydatetime(), len(rows))
                    for i in df[null_mask].index:
                        df.at[i, "temperature_c"] = float(fallback.iloc[i]["temp_c"])
                df["temperature_c"] = df["temperature_c"].astype(float)
                return df
        except Exception as e:
            logger.error(f"Failed to fetch history: {e}")
            return None

    async def _get_future_temps(self, start_time: datetime, hours: int) -> List[float]:
        df = await self.weather_service.get_forecast_from(start_time, hours)
        return [float(v) for v in df["temp_c"].values]

    def set_manual_temperature(self, temps_c: List[float]):
        self.weather_service.set_manual_override(temps_c)

    def clear_manual_temperature(self):
        self.weather_service.clear_manual_override()

    async def feedback_actuals(self, actual_mw: List[float], forecast_mw: Optional[List[float]] = None,
                                forecast_date: Optional[date] = None):
        """Feed actuals back to TIDE corrector and record metrics.

        If forecast_mw is not provided, tries to look up the cached forecast.
        """
        if not self.engine or len(actual_mw) != 24:
            return False
        dt = forecast_date or date.today()
        if forecast_mw is None or len(forecast_mw) != 24:
            cached = await self._cache_get(f"dlinear:24h:{dt.isoformat()}")
            if cached and len(cached.get("forecast_mw", [])) == 24:
                forecast_mw = cached["forecast_mw"]
            else:
                logger.warning(f"No cached forecast for {dt} — can't feed TIDE")
                return False
        self.engine.update(np.array(actual_mw), np.array(forecast_mw))
        await self.metrics.record(dt, "24h", forecast_mw, actual_mw)
        logger.info(f"TIDE fed with actuals for {dt} — MAE: {float(np.mean(np.abs(np.array(forecast_mw) - np.array(actual_mw)))):.1f} MW")
        return True

    async def forecast_dispatch(self, target_date_str: Optional[str] = None, user_temps: Optional[List[float]] = None) -> Dict[str, Any]:
        dt = date.fromisoformat(target_date_str) if target_date_str else date.today() + timedelta(days=1)
        return await self.forecast_for_date(dt, user_temps=user_temps)

    async def forecast_tomorrow(self, force_refresh: bool = False) -> Dict[str, Any]:
        return await self.forecast_for_date(date.today() + timedelta(days=1), force_refresh=force_refresh)

    async def forecast_for_date(self, target_date: date, force_refresh: bool = False, user_temps: Optional[List[float]] = None) -> Dict[str, Any]:
        cache_key = f"dlinear:24h:{target_date.isoformat()}"
        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached:
                return cached

        if self.engine is None or not self.engine.is_fitted:
            return {"error": "Engine not initialised", "forecast_mw": [0.0] * 24}

        history_df = await self._fetch_history()
        if history_df is None:
            return {"error": "Insufficient history", "forecast_mw": [0.0] * 24}

        last_ts = pd.to_datetime(history_df['date'].iloc[-1])
        start_time = last_ts + timedelta(hours=1)
        if user_temps is not None and len(user_temps) == 24:
            future_temps = user_temps
        else:
            future_temps = await self._get_future_temps(start_time, 24)

        prediction = self.engine.predict(history_df, horizon_hours=24, future_temps_c=future_temps)

        result = {
            "forecast_date": target_date.isoformat(),
            "forecast_mw": prediction.get("forecast_mw", [0.0] * 24),
            "p10_mw": prediction.get("p10_mw"),
            "p90_mw": prediction.get("p90_mw"),
            "uncertainty_mw": prediction.get("uncertainty_mw"),
            "temperature_c": future_temps,
            "engine": prediction.get("engine", "unknown"),
            "inference_ms": prediction.get("inference_ms"),
        }
        await self._cache_set(cache_key, "24h", result)
        await self.metrics.record(target_date, "24h", result["forecast_mw"])
        return result

    async def _forecast_day(self, history_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[float]]:
        last_ts = history_df['date'].iloc[-1]
        start_time = last_ts + timedelta(hours=1)
        future_temps = await self._get_future_temps(start_time, 24)
        pred = self.engine.predict(history_df, horizon_hours=24, future_temps_c=future_temps, use_tide=False)
        fw = np.array(pred.get("forecast_mw", [0.0] * 24))
        p10 = np.array(pred.get("p10_mw", fw.tolist()))
        p90 = np.array(pred.get("p90_mw", fw.tolist()))
        unc = np.array(pred.get("uncertainty_mw", [0.0] * 24))

        # Append forecast as fake history for next day's recursion
        last_ts = history_df['date'].iloc[-1]
        new_rows = pd.DataFrame({
            "date": [last_ts + timedelta(hours=h + 1) for h in range(24)],
            "demand_mw": fw,
            "temperature_c": future_temps,
        })
        history_df = pd.concat([history_df, new_rows], ignore_index=True).iloc[-(HISTORY_NEEDED + 24):]
        return fw, p10, p90, unc, history_df, future_temps

    async def forecast_multi_day(self, days: int) -> Dict[str, Any]:
        if self.engine is None or not self.engine.is_fitted:
            return {"error": "Engine not initialised", "hourly_mw": [], "daily_aggregates": []}

        history_df = await self._fetch_history()
        if history_df is None:
            return {"hourly_mw": [], "daily_aggregates": []}

        all_hours = []
        all_p10 = []
        all_p90 = []
        all_unc = []
        all_temps = []
        for d in range(days):
            fw, p10, p90, unc, history_df, temps = await self._forecast_day(history_df)
            all_hours.extend(fw.tolist())
            all_p10.extend(p10.tolist())
            all_p90.extend(p90.tolist())
            all_unc.extend(unc.tolist())
            all_temps.extend(temps)

        daily_aggs = []
        for d in range(days):
            offset = d * 24
            day_hours = all_hours[offset:offset + 24]
            daily_aggs.append({
                "date": (date.today() + timedelta(days=1 + d)).isoformat(),
                "peak_mw": round(float(max(day_hours)), 2),
                "mean_mw": round(float(np.mean(day_hours)), 2),
                "min_mw": round(float(min(day_hours)), 2),
                "total_energy_mwh": round(float(sum(day_hours)), 2),
            })

        return {"hourly_mw": all_hours, "p10_mw": all_p10, "p90_mw": all_p90, "uncertainty_mw": all_unc, "temperature_c": all_temps, "daily_aggregates": daily_aggs}

    async def forecast_7day(self, force_refresh: bool = False) -> Dict[str, Any]:
        cache_key = f"dlinear:7d:{date.today().isoformat()}"
        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached:
                return cached
        result = await self.forecast_multi_day(7)
        payload = {
            "forecast_date": date.today().isoformat(),
            "hourly_mw": result.get("hourly_mw", []),
            "p10_mw": result.get("p10_mw", []),
            "p90_mw": result.get("p90_mw", []),
            "uncertainty_mw": result.get("uncertainty_mw", []),
            "temperature_c": result.get("temperature_c", []),
            "daily_aggregates": result.get("daily_aggregates", []),
        }
        await self._cache_set(cache_key, "7d", payload)
        return payload

    async def forecast_30day(self, force_refresh: bool = False) -> Dict[str, Any]:
        cache_key = f"dlinear:30d:{date.today().isoformat()}"
        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached:
                return cached
        result = await self.forecast_multi_day(30)
        payload = {
            "forecast_date": date.today().isoformat(),
            "daily_aggregates": result.get("daily_aggregates", []),
        }
        await self._cache_set(cache_key, "30d", payload)
        return payload

    async def forecast_90day(self, force_refresh: bool = False) -> Dict[str, Any]:
        cache_key = f"dlinear:90d:{date.today().isoformat()}"
        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached:
                return cached
        result = await self.forecast_multi_day(90)
        aggs = result.get("daily_aggregates", [])
        weekly = []
        for i in range(0, len(aggs), 7):
            chunk = aggs[i:i + 7]
            if not chunk:
                continue
            peak = max(d["peak_mw"] for d in chunk)
            mean = float(np.mean([d["mean_mw"] for d in chunk]))
            mn = min(d["min_mw"] for d in chunk)
            total = sum(d["total_energy_mwh"] for d in chunk)
            weekly.append({
                "week_start": chunk[0]["date"],
                "week_end": chunk[-1]["date"],
                "mean_mw": round(mean, 2),
                "peak_mw": round(peak, 2),
                "min_mw": round(mn, 2),
                "total_energy_mwh": round(total, 2),
            })
        payload = {
            "forecast_date": date.today().isoformat(),
            "weekly_aggregates": weekly,
        }
        await self._cache_set(cache_key, "90d", payload)
        return payload
