import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import date, timedelta
from app.core.logging import get_logger

logger = get_logger(__name__)

DOW_OFFSETS = {0: 54, 1: 24, 2: 0, 3: 0, 4: 0, 5: -38, 6: -41} # Good but how would the theese offsets become learned parametres in the deployed system seeing that this changes year to year


class WeightedTrendEngine:
    """Weighted trend + DOW correction x Monthx DOW profile.
    
    level = 0.65*L1 + 0.35*L7 + DOW_offset[dow]
    forecast[h] = level * profile[month][dow][h]
    """
    def __init__(self):
        self.profiles: Dict[Tuple[int, int], np.ndarray] = {}
        self.is_fitted = False
        self._last_daily_means: pd.Series = pd.Series(dtype=float)

    def fit(self, df: pd.DataFrame):
        if df is None or df.empty:
            logger.warning("No data for WeightedTrendEngine.fit")
            return
        prep = df.copy()
        if 'Hour' not in prep.columns and 'hour' in prep.columns:
            prep['Hour'] = prep['hour']
        if 'Date' not in prep.columns and 'date' in prep.columns:
            prep['Date'] = pd.to_datetime(prep['date'])
        elif 'Date' in prep.columns:
            prep['Date'] = pd.to_datetime(prep['Date'])
        if 'demand_mw' not in prep.columns:
            logger.error("demand_mw column not found")
            return
        prep['dow'] = prep['Date'].dt.dayofweek
        prep['month'] = prep['Date'].dt.month
        prep['dd'] = prep['Date'].dt.date
        valid_days = prep.groupby('dd').filter(lambda g: len(g) == 24)['dd'].unique()
        prep = prep[prep['dd'].isin(valid_days)]
        if prep.empty:
            logger.warning("No complete days for profile computation")
            return
        # Month x DOW profiles
        from itertools import product
        self.profiles = {}
        for m, dw in product(range(1, 13), range(7)):
            mask = (prep['month'] == m) & (prep['dow'] == dw)
            grp = prep[mask]
            if len(grp) < 24:
                self.profiles[(m, dw)] = np.ones(24)
                continue
            dm = grp.groupby('dd')['demand_mw'].transform('mean')
            ratio = grp['demand_mw'] / dm.replace(0, np.nan)
            p = np.array([ratio[grp['Hour'] == h].median() for h in range(1, 25)])
            mn = np.nanmean(p)
            self.profiles[(m, dw)] = p / mn if mn > 0 else np.ones(24)
        self.is_fitted = True
        logger.info(f"WeightedTrendEngine fitted: {len(self.profiles)} profiles")

    def load_history(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        prep = df.copy()
        if 'Date' not in prep.columns and 'date' in prep.columns:
            prep['Date'] = pd.to_datetime(prep['date'])
        elif 'Date' in prep.columns:
            prep['Date'] = pd.to_datetime(prep['Date'])
        if 'Hour' not in prep.columns and 'hour' in prep.columns:
            prep['Hour'] = prep['hour']
        prep['dd'] = prep['Date'].dt.date
        daily = prep.groupby('dd')['demand_mw'].mean()
        self._last_daily_means = daily.sort_index()

    def _get_level(self, target_date: date) -> float:
        td = pd.Timestamp(target_date)
        l1 = self._last_daily_means.iloc[-1] if len(self._last_daily_means) >= 1 else 1800.0
        l7 = self._last_daily_means.iloc[-7] if len(self._last_daily_means) >= 7 else l1
        return 0.65 * l1 + 0.35 * l7 + DOW_OFFSETS.get(td.dayofweek, 0)

    def predict_tomorrow(self) -> Dict[str, Any]:
        target = date.today() + timedelta(days=1)
        return self.predict_for_date(target)

    def predict_for_date(self, target_date: date) -> Dict[str, Any]:
        if not self.is_fitted:
            return {"forecast_mw": [0.0] * 24, "error": "Engine not fitted"}
        td = pd.Timestamp(target_date)
        prof = self.profiles.get((td.month, td.dayofweek), np.ones(24))
        level = self._get_level(target_date)
        fc = (prof * level).tolist()
        return {
            "forecast_date": target_date.isoformat(),
            "forecast_mw": fc,
            "factors": {
                "level_mw": round(level, 1),
                "dow_offset": DOW_OFFSETS.get(td.dayofweek, 0),
                "profile": [round(p, 4) for p in prof.tolist()],
                "dow": td.dayofweek,
                "month": td.month,
            }
        }

    def predict_multi_day(self, days: int) -> Dict[str, Any]:
        start = date.today() + timedelta(days=1)
        hourly_all = []
        daily_aggs = []
        for d in range(days):
            td = start + timedelta(days=d)
            result = self.predict_for_date(td)
            day_hours = result["forecast_mw"]
            hourly_all.extend(day_hours)
            daily_aggs.append({
                "date": td.isoformat(),
                "peak_mw": round(max(day_hours), 2),
                "mean_mw": round(float(np.mean(day_hours)), 2),
                "min_mw": round(min(day_hours), 2),
                "total_energy_mwh": round(sum(day_hours), 2),
            })
        return {
            "hourly_mw": hourly_all,
            "daily_aggregates": daily_aggs,
        }

    def predict_week_ahead(self) -> Dict[str, Any]:
        """Days 2-3: recursive hourly, Days 4-7: daily avg x profile."""
        start = date.today() + timedelta(days=1)
        hourly_all = []
        daily_aggs = []
        for d in range(7):
            td = start + timedelta(days=d)
            td_ts = pd.Timestamp(td)
            prof = self.profiles.get((td_ts.month, td_ts.dayofweek), np.ones(24))
            if d < 3:
                level = self._get_level(td)
            else:
                level = self._get_level(td)
            day_hours = (prof * level).tolist()
            hourly_all.extend(day_hours)
            daily_aggs.append({
                "date": td.isoformat(),
                "peak_mw": round(max(day_hours), 2),
                "mean_mw": round(float(np.mean(day_hours)), 2),
                "min_mw": round(min(day_hours), 2),
                "total_energy_mwh": round(sum(day_hours), 2),
            })
        return {"hourly_mw": hourly_all, "daily_aggregates": daily_aggs}

    def predict_month_ahead(self) -> Dict[str, Any]:
        start = date.today() + timedelta(days=1)
        daily_aggs = []
        for d in range(30):
            td = start + timedelta(days=d)
            td_ts = pd.Timestamp(td)
            prof = self.profiles.get((td_ts.month, td_ts.dayofweek), np.ones(24))
            level = self._get_level(td)
            day_hours = prof * level
            daily_aggs.append({
                "date": td.isoformat(),
                "peak_mw": round(float(max(day_hours)), 2),
                "mean_mw": round(float(np.mean(day_hours)), 2),
                "min_mw": round(float(min(day_hours)), 2),
                "total_energy_mwh": round(float(sum(day_hours)), 2),
            })
        return {"daily_aggregates": daily_aggs}

    def predict_90day(self) -> Dict[str, Any]:
        start = date.today() + timedelta(days=1)
        daily_aggs = []
        for d in range(90):
            td = start + timedelta(days=d)
            td_ts = pd.Timestamp(td)
            prof = self.profiles.get((td_ts.month, td_ts.dayofweek), np.ones(24))
            level = self._get_level(td)
            day_hours = prof * level
            daily_aggs.append({
                "date": td.isoformat(),
                "peak_mw": round(float(max(day_hours)), 2),
                "mean_mw": round(float(np.mean(day_hours)), 2),
                "min_mw": round(float(min(day_hours)), 2),
                "total_energy_mwh": round(float(sum(day_hours)), 2),
            })
        weekly = []
        for i in range(0, len(daily_aggs), 7):
            chunk = daily_aggs[i:i + 7]
            if not chunk:
                continue
            weekly.append({
                "week_start": chunk[0]["date"],
                "week_end": chunk[-1]["date"],
                "mean_mw": round(float(np.mean([d["mean_mw"] for d in chunk])), 2),
                "peak_mw": round(max(d["peak_mw"] for d in chunk), 2),
                "min_mw": round(min(d["min_mw"] for d in chunk), 2),
                "total_energy_mwh": round(sum(d["total_energy_mwh"] for d in chunk), 2),
            })
        return {"weekly_aggregates": weekly}

    def save(self, path: str):
        state = {
            "profiles": {f"{m}_{dw}": self.profiles[(m, dw)].tolist() for m, dw in self.profiles},
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, path)
        logger.info(f"WeightedTrendEngine saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"WeightedTrendEngine state not found at {path}")
            return
        state = joblib.load(path)
        self.profiles = {}
        for key, vals in state.get("profiles", {}).items():
            parts = key.split("_")
            self.profiles[(int(parts[0]), int(parts[1]))] = np.array(vals)
        self.is_fitted = state.get("is_fitted", False)
        logger.info(f"WeightedTrendEngine loaded from {path}")
