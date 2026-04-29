import os
import numpy as np
import pandas as pd
import joblib
import datetime
from typing import Dict, Any
from statsmodels.tsa.holtwinters import Holt
from scipy.optimize import minimize_scalar

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class TrendModel:
    def __init__(self):
        self.model = None
        self.last_train_date = None
        self.last_val = 0.0
        self.cap_growth = 1.12 # caping growth should't that be a learned parameter?

    def fit(self, daily_mean: pd.Series):
        if len(daily_mean) < 2:
            logger.warning("Not enough daily data for Trend fitting")
            return
            
        self.model = Holt(daily_mean.values, exponential=False, damped_trend=True).fit(
            optimized=True, use_brute=False)
        self.last_train_date = daily_mean.index[-1]
        self.last_val = self.model.fittedvalues[-1]

    def nudge_trend(self, new_val: float, anchor_date: datetime.date):
        """
        Manually shift the trend baseline to match recent SCADA reality.
        """
        logger.info(f"Nudging Trend baseline from {self.last_val:.1f} to {new_val:.1f} MW")
        self.last_val = new_val
        self.last_train_date = anchor_date
        
    def get_trend_array(self, dates_series: pd.Series) -> np.ndarray:
        if self.model is None:
            return np.full(len(dates_series), self.last_val if self.last_val > 0 else 80.0)
            
        days_ahead = np.array([
            (pd.Timestamp(d) - pd.Timestamp(self.last_train_date)).days
            for d in dates_series
        ])
        
        max_ahead = int(days_ahead.max()) + 1
        if max_ahead > 0:
            future = np.array(self.model.forecast(max_ahead))
        else:
            future = np.array([self.last_val])
            
        full = np.concatenate([[self.last_val], future])
        # Growth cap
        cap = self.last_val * (self.cap_growth ** (np.arange(len(full)) / 365.25))
        full = np.clip(full, 0, cap)
        
        idx = np.clip(days_ahead, 0, len(full) - 1)
        return full[idx]

class SeasonalModel:
    def __init__(self):
        self.s_ts = np.ones(96)
        self.s_dow = np.ones(7)

    def fit(self, df_clean: pd.DataFrame):
        # S_ts: median of (load / daily_mean) per 15-min slot
        daily_m = df_clean.groupby('Date')['Masked_Load'].mean()
        df_clean['DailyMean'] = df_clean['Date'].map(daily_m)
        df_clean = df_clean[df_clean['DailyMean'] > 1.0]
        
        if df_clean.empty:
            return

        df_clean['Ratio_ts'] = df_clean['Masked_Load'] / df_clean['DailyMean']
        self.s_ts = df_clean.groupby('TimeSlot')['Ratio_ts'].median().values
        if len(self.s_ts) < 96:
            # Pad if missing slots
            full_ts = np.ones(96)
            for i, v in enumerate(self.s_ts):
                full_ts[i] = v
            self.s_ts = full_ts
        self.s_ts = self.s_ts / self.s_ts.mean()

        # S_dow: median of (daily_mean / overall_mean) per day-of-week
        dow_mean = df_clean.groupby('DOW')['DailyMean'].median()
        overall_mean = daily_m.mean()
        dow_factor = (dow_mean / overall_mean).reindex(range(7)).fillna(1.0).values
        self.s_dow = dow_factor / dow_factor.mean()

    def apply(self, ts_arr: np.ndarray, dow_arr: np.ndarray) -> np.ndarray:
        ts_arr = np.array(ts_arr) % 96
        dow_arr = np.array(dow_arr) % 7
        return self.s_ts[ts_arr] * self.s_dow[dow_arr]

class TemperatureModel:
    def __init__(self, knot=24.0):
        self.knot = knot
        self.theta = np.array([0.0, 0.0, 1.0]) # [slope_low, slope_high, intercept]

    def fit(self, temp: np.ndarray, ratio_target: np.ndarray):
        mask = np.isfinite(ratio_target) & np.isfinite(temp) & (ratio_target > 0.1) & (ratio_target < 5.0)
        T, R = temp[mask], ratio_target[mask]
        
        if len(T) < 10:
            return

        def sse(k):
            X1 = np.minimum(T - k, 0)
            X2 = np.maximum(T - k, 0)
            X = np.column_stack([X1, X2, np.ones_like(T)])
            theta, *_ = np.linalg.lstsq(X, R, rcond=None)
            return np.mean((R - X @ theta) ** 2)

        res = minimize_scalar(sse, bounds=(20.0, 30.0), method='bounded')
        self.knot = res.x
        X1 = np.minimum(T - self.knot, 0)
        X2 = np.maximum(T - self.knot, 0)
        X = np.column_stack([X1, X2, np.ones_like(T)])
        self.theta, *_ = np.linalg.lstsq(X, R, rcond=None)

    def apply(self, temp_arr: np.ndarray) -> np.ndarray:
        T = np.asarray(temp_arr, dtype=float)
        X1 = np.minimum(T - self.knot, 0)
        X2 = np.maximum(T - self.knot, 0)
        return self.theta[0] * X1 + self.theta[1] * X2 + self.theta[2]

class HolidayModel:
    def __init__(self):
        self.profile = np.ones(96)

    def fit(self, ts_arr: np.ndarray, is_holiday_arr: np.ndarray, ratio_arr: np.ndarray):
        df_h = pd.DataFrame({
            'ts': np.array(ts_arr) % 96,
            'hol': np.array(is_holiday_arr),
            'r': ratio_arr
        })
        hol = df_h[df_h['hol'] == 1].groupby('ts')['r'].median().reindex(range(96)).fillna(np.nan)
        norm = df_h[df_h['hol'] == 0].groupby('ts')['r'].median().reindex(range(96)).fillna(np.nan)
        supp = (hol / norm).fillna(1.0).clip(0.5, 1.2).values
        self.profile = supp

    def apply(self, ts_arr: np.ndarray, is_holiday_arr: np.ndarray) -> np.ndarray:
        ts_arr = np.array(ts_arr) % 96
        return np.where(is_holiday_arr == 1, self.profile[ts_arr], 1.0)

class KalmanBiasCorrector:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.bias = 0.0

    def update(self, actual: float, forecast: float):
        if np.isfinite(actual) and np.isfinite(forecast):
            err = actual - forecast
            self.bias = self.alpha * err + (1 - self.alpha) * self.bias

    def correct(self, raw: np.ndarray) -> np.ndarray:
        return raw + self.bias

class DecomEngine:
    def __init__(self):
        self.trend = TrendModel()
        self.seasonal = SeasonalModel()
        self.temp = TemperatureModel(knot=settings.TEMP_KNOT)
        self.holiday = HolidayModel()
        self.kalman = KalmanBiasCorrector()
        self.is_fitted = False

    def save(self, path: str):
        state = {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "temp": self.temp,
            "holiday": self.holiday,
            "is_fitted": self.is_fitted
        }
        joblib.dump(state, path)
        logger.info(f"DecomEngine saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"DecomEngine state dict not found at {path}")
            return
        state = joblib.load(path)
        self.trend = state["trend"]
        self.seasonal = state["seasonal"]
        self.temp = state["temp"]
        self.holiday = state["holiday"]
        self.is_fitted = state["is_fitted"]
        logger.info(f"DecomEngine loaded from {path}")

    def predict(self, df: pd.DataFrame, horizon_steps: int = 96) -> Dict[str, Any]:
        """
        Generate forecast and components.
        df should contain 'DATETIME', 'Temp', 'Is_Holiday', 'precip_mm', etc.
        """
        # 1. Structural components
        t_arr = self.trend.get_trend_array(df['Date'])
        s_arr = self.seasonal.apply(df['TimeSlot'].values, df['DOW'].values)
        tmp_arr = self.temp.apply(df['Temp'].values)
        h_arr = self.holiday.apply(df['TimeSlot'].values, df['Is_Holiday'].values)
        
        # 2. Advanced Physics Adjustments (from Operator Interview)
        
        # A: Rain Suppressor (Rain = less AC + less heating devices)
        # Suppress load by up to 15% during heavy rain (> 5mm)
        precip = df['precip_mm'].values if 'precip_mm' in df.columns else np.zeros(len(df))
        rain_mult = 1.0 - (np.clip(precip, 0, 5) / 5.0) * 0.15
        
        # B: Line Efficiency Gain (Cooler temp = lower transmission losses)
        # Gain of ~1-2% when temp is significantly below average comfort (22C)
        temp_vals = df['Temp'].values
        efficiency_gain = np.where(temp_vals < 22.0, 0.985, 1.0) # 1.5% reduction in 'demand' due to efficiency
        
        raw_struct = t_arr * s_arr * tmp_arr * h_arr * rain_mult * efficiency_gain
        
        # 3. Kalman correction (for short-term)
        corrected = self.kalman.correct(raw_struct)
        
        return {
            "forecast_mw": corrected.tolist(),
            "components": {
                "trend": (t_arr * s_arr).tolist(), 
                "temp_effect": (t_arr * s_arr * (tmp_arr - 1.0)).tolist(),
                "holiday_effect": (t_arr * s_arr * tmp_arr * (h_arr - 1.0)).tolist(),
                "rain_impact": (t_arr * s_arr * tmp_arr * h_arr * (rain_mult - 1.0)).tolist(),
                "efficiency_gain": (t_arr * s_arr * tmp_arr * h_arr * rain_mult * (efficiency_gain - 1.0)).tolist(),
                "kalman_bias": [self.kalman.bias] * len(df)
            },
            "factors": {
                "trend_mw": t_arr.tolist(),
                "seasonal_ratio": s_arr.tolist(),
                "temp_ratio": tmp_arr.tolist(),
                "holiday_ratio": h_arr.tolist(),
                "rain_ratio": rain_mult.tolist(),
                "efficiency_ratio": efficiency_gain.tolist()
            }
        }
