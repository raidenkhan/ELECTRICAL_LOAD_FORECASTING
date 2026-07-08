from typing import Dict, Any, Optional
import io, csv
import pandas as pd
import numpy as np
import os
from datetime import date, timedelta, datetime, timezone
from sqlalchemy import select, delete
from fastapi import UploadFile

from app.core.logging import get_logger
from app.ml.weighted_trend_engine import WeightedTrendEngine
from app.db.models.ecg_history import EcgHistoricalDemand
from app.db.models.forecast_cache import ForecastCache
from app.db.session import AsyncSessionLocal
import joblib

logger = get_logger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'models', 'weighted_trend_engine.joblib'
)
MAX_MODEL_AGE_DAYS = 7


class BaselineForecastService:
    def __init__(self):
        self.engine: Optional[WeightedTrendEngine] = None
        self._load_engine()

    def _load_engine(self):
        self.engine = WeightedTrendEngine()
        self.engine.load(MODEL_PATH)
        if not self.engine.is_fitted:
            logger.info("WeightedTrendEngine not loaded from disk, will fit from DB history")

    def _model_age_days(self) -> Optional[int]:
        if not os.path.exists(MODEL_PATH):
            return None
        mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        return (datetime.now() - mtime).days

    def _auto_retrain_if_stale(self, df: pd.DataFrame):
        age = self._model_age_days()
        if age is None or age > MAX_MODEL_AGE_DAYS:
            logger.info(f"WT+DOW model age: {age} days — auto-retraining")
            self.engine = WeightedTrendEngine()
            self.engine.fit(df)
            self.engine.save(MODEL_PATH)
            logger.info("WT+DOW model retrained and saved")

    async def _ensure_fitted(self):
        if self.engine is not None and self.engine.is_fitted and len(self.engine._last_daily_means) > 0:
            return
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(EcgHistoricalDemand).order_by(EcgHistoricalDemand.date.desc(), EcgHistoricalDemand.hour.desc()).limit(28 * 24)
                result = await db.execute(stmt)
                rows = result.scalars().all()
                if not rows:
                    logger.warning("No historical data for BaselineForecastService")
                    return
                df = pd.DataFrame([{
                    'Date': r.date,
                    'Hour': r.hour,
                    'demand_mw': r.demand_mw,
                } for r in rows])
                df['Date'] = pd.to_datetime(df['Date'])
                if not self.engine.is_fitted:
                    self.engine.fit(df)
                    self._auto_retrain_if_stale(df)
                self.engine.load_history(df)
        except Exception as e:
            logger.error(f"Failed to load history: {e}")

    async def _cache_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(ForecastCache).where(ForecastCache.cache_key == cache_key)
                result = await db.execute(stmt)
                entry = result.scalar_one_or_none()
                if entry:
                    return dict(entry.data)
        except Exception:
            pass
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
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    async def _cache_clear_all(self):
        try:
            async with AsyncSessionLocal() as db:
                await db.execute(delete(ForecastCache))
                await db.commit()
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")

    async def data_freshness_info(self) -> Dict[str, Any]:
        await self._ensure_fitted()
        if self.engine is None or len(self.engine._last_daily_means) == 0:
            return {"latest_date": "N/A", "days_stale": -1, "status": "unknown"}
        latest = self.engine._last_daily_means.index[-1]
        today = date.today()
        days_stale = (today - latest).days
        status = "fresh" if days_stale <= 2 else ("stale" if days_stale <= 14 else "old")
        return {
            "latest_date": latest.isoformat(),
            "days_stale": days_stale,
            "status": status,
        }

    async def quick_refit(self, upload_file: UploadFile) -> Dict[str, Any]:
        content = await upload_file.read()
        text = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return {"status": "error", "detail": "Empty CSV"}
        df = pd.DataFrame(rows)
        for col in ['date', 'hour', 'demand_mw']:
            if col not in df.columns:
                return {"status": "error", "detail": f"Missing column: {col}"}
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['hour'].astype(int)
        df['demand_mw'] = df['demand_mw'].astype(float)
        df['dd'] = df['date'].dt.date
        day_counts = df.groupby('dd').size()
        incomplete = day_counts[day_counts != 24]
        if len(incomplete) > 0:
            return {"status": "error", "detail": f"Incomplete days found: {list(incomplete.index)}"}
        self.engine.load_history(df)
        await self._cache_clear_all()
        result = self.engine.predict_tomorrow()
        days_loaded = len(day_counts)
        latest_date = df['dd'].max().isoformat()
        freshness = await self.data_freshness_info()
        # Extract latest day's 24 actual values for TIDE feedback
        last_day = df[df['dd'] == latest_date].sort_values('hour')['demand_mw'].tolist()
        return {
            "status": "ok",
            "records_loaded": len(df),
            "latest_date": latest_date,
            "days_loaded": days_loaded,
            "forecast": result,
            "freshness": freshness,
            "_actual_mw": last_day,
        }

    async def forecast_tomorrow(self, force_refresh: bool = False) -> Dict[str, Any]:
        cache_key = f"baseline_24h:{date.today().isoformat()}"
        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached:
                return cached
        await self._ensure_fitted()
        if self.engine is None or not self.engine.is_fitted:
            return {"error": "Engine not trained", "forecast_mw": [0.0] * 24}
        result = self.engine.predict_tomorrow()
        await self._cache_set(cache_key, "24h", result)
        return result

    async def forecast_for_date(self, target_date: date, force_refresh: bool = False) -> Dict[str, Any]:
        cache_key = f"baseline_24h:{target_date.isoformat()}"
        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached:
                return cached
        await self._ensure_fitted()
        if self.engine is None or not self.engine.is_fitted:
            return {"error": "Engine not trained", "forecast_mw": [0.0] * 24}
        result = self.engine.predict_for_date(target_date)
        await self._cache_set(cache_key, "24h", result)
        return result

    async def forecast_7day(self, force_refresh: bool = False) -> Dict[str, Any]:
        cache_key = f"baseline_7d:{date.today().isoformat()}"
        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached:
                return cached
        await self._ensure_fitted()
        if self.engine is None or not self.engine.is_fitted:
            return {"forecast_date": date.today().isoformat(), "hourly_mw": [], "daily_aggregates": []}
        result = self.engine.predict_week_ahead()
        payload = {"forecast_date": date.today().isoformat(), **result}
        await self._cache_set(cache_key, "7d", payload)
        return payload

    async def forecast_30day(self, force_refresh: bool = False) -> Dict[str, Any]:
        cache_key = f"baseline_30d:{date.today().isoformat()}"
        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached:
                return cached
        await self._ensure_fitted()
        if self.engine is None or not self.engine.is_fitted:
            return {"forecast_date": date.today().isoformat(), "daily_aggregates": []}
        result = self.engine.predict_month_ahead()
        payload = {"forecast_date": date.today().isoformat(), **result}
        await self._cache_set(cache_key, "30d", payload)
        return payload

    async def forecast_90day(self, force_refresh: bool = False) -> Dict[str, Any]:
        cache_key = f"baseline_90d:{date.today().isoformat()}"
        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached:
                return cached
        await self._ensure_fitted()
        if self.engine is None or not self.engine.is_fitted:
            return {"forecast_date": date.today().isoformat(), "weekly_aggregates": []}
        result = self.engine.predict_90day()
        payload = {"forecast_date": date.today().isoformat(), **result}
        await self._cache_set(cache_key, "90d", payload)
        return payload
