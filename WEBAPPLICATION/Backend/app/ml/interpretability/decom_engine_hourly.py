import os
import numpy as np
import pandas as pd
import joblib
import datetime
from typing import Dict, Any, Optional
from statsmodels.tsa.holtwinters import Holt
from scipy.optimize import minimize_scalar
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models.ecg_history import EcgHistoricalDemand

logger = get_logger(__name__)


class TrendModel:
    def __init__(self):
        self.model = None
        self.last_train_date = None
        self.last_val = 0.0
        self.daily_smoothed = {}

    def fit(self, daily_mean: pd.Series):
        if len(daily_mean) < 2:
            logger.warning("Not enough daily data for Trend fitting")
            return
        self.model = Holt(daily_mean.values, exponential=False, damped_trend=True).fit(
            optimized=True, use_brute=False)
        self.last_train_date = daily_mean.index[-1]
        self.last_val = self.model.fittedvalues[-1]

        smooth = daily_mean.rolling(14, center=True, min_periods=1).mean().fillna(method='bfill').fillna(method='ffill')
        for dt, val in smooth.items():
            d = dt.date() if hasattr(dt, 'date') else dt
            self.daily_smoothed[d] = val

    def nudge_trend(self, new_val: float, anchor_date: datetime.date):
        logger.info(f"Nudging Trend baseline from {self.last_val:.1f} to {new_val:.1f} MW")
        self.last_val = new_val
        self.last_train_date = anchor_date

    def get_trend_array(self, dates_series: pd.Series) -> np.ndarray:
        result = []
        for d in dates_series:
            dt_val = pd.Timestamp(d)
            day = dt_val.date()
            if day in self.daily_smoothed:
                result.append(self.daily_smoothed[day])
            else:
                days_ahead = (dt_val - pd.Timestamp(self.last_train_date)).days
                if self.model is not None and days_ahead >= 0:
                    fut = np.array(self.model.forecast(days_ahead + 1))
                    result.append(fut[-1])
                else:
                    result.append(self.last_val if self.last_val > 0 else 1800.0)
        return np.array(result)


class SeasonalModel:
    def __init__(self):
        self.s_ts = np.ones(24)
        self.s_dow = np.ones(7)

    def fit(self, df_clean: pd.DataFrame):
        daily_m = df_clean.groupby('Date')['demand_mw'].mean()
        df_clean = df_clean.copy()
        df_clean['DailyMean'] = df_clean['Date'].map(daily_m)
        df_clean = df_clean[df_clean['DailyMean'] > 1.0]
        if df_clean.empty:
            return
        df_clean['Ratio_ts'] = df_clean['demand_mw'] / df_clean['DailyMean']
        self.s_ts = df_clean.groupby('Hour')['Ratio_ts'].median().values
        if len(self.s_ts) < 24:
            full_ts = np.ones(24)
            for i, v in enumerate(self.s_ts):
                full_ts[i] = v
            self.s_ts = full_ts
        self.s_ts = self.s_ts / self.s_ts.mean()

        dow_mean = df_clean.groupby('DOW')['DailyMean'].median()
        overall_mean = daily_m.mean()
        dow_factor = (dow_mean / overall_mean).reindex(range(7)).fillna(1.0).values
        self.s_dow = dow_factor / dow_factor.mean()

    def apply(self, hour_arr: np.ndarray, dow_arr: np.ndarray) -> np.ndarray:
        hour_idx = (np.array(hour_arr) - 1) % 24
        dow_idx = np.array(dow_arr) % 7
        return self.s_ts[hour_idx] * self.s_dow[dow_idx]


class TemperatureModel:
    def __init__(self, knot=0.0):
        self.knot = knot
        self.theta = np.array([0.0, 0.0, 1.0])
        self.baseline_ratio = 1.0
        self.temp_means_by_hour = np.zeros(24)

    def fit(self, temp: np.ndarray, ratio_target: np.ndarray, hour_arr: np.ndarray):
        mask = np.isfinite(ratio_target) & np.isfinite(temp) & (ratio_target > 0.1) & (ratio_target < 5.0)
        T, R, H = temp[mask], ratio_target[mask], hour_arr[mask]
        if len(T) < 10:
            return

        # Compute mean temp per hour to deseasonalize
        for h in range(1, 25):
            h_mask = H == h
            if h_mask.sum() > 0:
                self.temp_means_by_hour[h-1] = float(np.mean(T[h_mask]))
        # Use anomalies: deviation from expected temp for that hour
        T_anom = T - self.temp_means_by_hour[(H - 1).astype(int)]

        def sse(k):
            X1 = np.minimum(T_anom - k, 0)
            X2 = np.maximum(T_anom - k, 0)
            X = np.column_stack([X1, X2, np.ones_like(T)])
            theta, *_ = np.linalg.lstsq(X, R, rcond=None)
            return np.mean((R - X @ theta) ** 2)

        res = minimize_scalar(sse, bounds=(-5.0, 5.0), method='bounded')
        self.knot = res.x
        X1 = np.minimum(T_anom - self.knot, 0)
        X2 = np.maximum(T_anom - self.knot, 0)
        X = np.column_stack([X1, X2, np.ones_like(T)])
        self.theta, *_ = np.linalg.lstsq(X, R, rcond=None)

        fitted = X @ self.theta
        self.baseline_ratio = float(np.mean(fitted))

    def apply(self, temp_arr: np.ndarray, hour_arr: np.ndarray) -> np.ndarray:
        T = np.asarray(temp_arr, dtype=float)
        H = np.asarray(hour_arr, dtype=int)
        T_anom = T - self.temp_means_by_hour[(H - 1) % 24]
        X1 = np.minimum(T_anom - self.knot, 0)
        X2 = np.maximum(T_anom - self.knot, 0)
        raw = self.theta[0] * X1 + self.theta[1] * X2 + self.theta[2]
        return raw / self.baseline_ratio


class HolidayModel:
    def __init__(self):
        self.profile = np.ones(24)

    def fit(self, hour_arr: np.ndarray, is_holiday_arr: np.ndarray, ratio_arr: np.ndarray):
        df_h = pd.DataFrame({
            'hour': np.array(hour_arr),
            'hol': np.array(is_holiday_arr),
            'r': ratio_arr,
        })
        hol = df_h[df_h['hol'] == 1].groupby('hour')['r'].median().reindex(range(1, 25)).fillna(np.nan)
        norm = df_h[df_h['hol'] == 0].groupby('hour')['r'].median().reindex(range(1, 25)).fillna(np.nan)
        supp = (hol / norm).fillna(1.0).clip(0.5, 1.2).values
        self.profile = supp

    def apply(self, hour_arr: np.ndarray, is_holiday_arr: np.ndarray) -> np.ndarray:
        hour_idx = (np.array(hour_arr) - 1) % 24
        return np.where(np.array(is_holiday_arr) == 1, self.profile[hour_idx], 1.0)


class GrowthEngine:
    def __init__(self, annual_growth: float = 0.08):
        self.annual_growth = annual_growth
        self.base_year_mean = 1900.0
        self.base_date = None
        self.baseline_mult = 1.0

    def fit_from_history(self, daily_means: pd.Series):
        if len(daily_means) < 60:
            logger.info("Not enough history for growth computation, using default 8%")
            return
        self.base_date = daily_means.index[0].date() if hasattr(daily_means.index[0], 'date') else daily_means.index[0]
        self.base_year_mean = daily_means.iloc[:30].mean()

        if len(daily_means) > 360:
            recent_mean = daily_means.iloc[-90:].mean()
            last_d = daily_means.index[-1].date() if hasattr(daily_means.index[-1], 'date') else daily_means.index[-1]
            days_span = (last_d - self.base_date).days
            if days_span > 180:
                implied_growth = (recent_mean / self.base_year_mean) ** (365.25 / days_span) - 1
                self.annual_growth = max(0.01, min(0.20, implied_growth))
                logger.info(f"Computed YoY growth: {self.annual_growth*100:.1f}%")

        # Compute baseline: mean growth_mult over the training period
        mults = []
        for d in daily_means.index:
            dt = d.date() if hasattr(d, 'date') else d
            mults.append(self.get_growth_mult(dt))
        self.baseline_mult = float(np.mean(mults)) if mults else 1.0

    def get_growth_mult(self, target_date: datetime.date) -> float:
        if self.base_date is None:
            return 1.0
        days_diff = (target_date - self.base_date).days
        years = days_diff / 365.25
        return 1.0 + years * self.annual_growth

    def get_growth_mult(self, target_date: datetime.date) -> float:
        if self.base_date is None:
            return 1.0
        days_diff = (target_date - self.base_date).days
        years = days_diff / 365.25
        return 1.0 + years * self.annual_growth


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


class AR1Corrector:
    """AR(1) residual correction: e_t = rho * e_{t-1} + noise.
    
    During forecasting, the correction cascades: forecast_h = structural_h + rho^h * last_residual.
    During online operation, feed actual errors via update() to keep last_residual current.
    """
    def __init__(self):
        self.rho = 0.0
        self.last_residual = 0.0
        self.is_fitted = False

    def fit(self, residuals: np.ndarray):
        mask = np.isfinite(residuals)
        r = residuals[mask]
        if len(r) < 10:
            logger.warning("AR(1): too few finite residuals to fit")
            return
        e_t = r[1:]
        e_t_1 = r[:-1]
        # OLS through origin: e_t = rho * e_{t-1}
        denom = np.sum(e_t_1 * e_t_1)
        if denom < 1e-10:
            logger.warning("AR(1): denominator too small, skipping")
            return
        self.rho = float(np.clip(np.sum(e_t_1 * e_t) / denom, -0.99, 0.99))
        self.last_residual = float(r[-1])
        self.is_fitted = True
        logger.info(f"AR(1) fitted: rho={self.rho:.4f}, last_residual={self.last_residual:.1f}")

    def correct(self, structural: np.ndarray) -> np.ndarray:
        if not self.is_fitted or abs(self.rho) < 1e-6:
            return structural.copy()
        corrected = structural.copy()
        res = float(self.last_residual)
        for i in range(len(corrected)):
            corrected[i] = corrected[i] + self.rho * res
            res = self.rho * res
        return corrected

    def update(self, actual: float, forecast: float):
        if np.isfinite(actual) and np.isfinite(forecast):
            self.last_residual = actual - forecast


RESIDUAL_HISTORY_HOURS = 720


class DecomEngineHourly:
    def __init__(self):
        self.trend = TrendModel()
        self.seasonal = SeasonalModel()
        self.temp = TemperatureModel(knot=0.0)
        self.holiday = HolidayModel()
        self.growth = GrowthEngine()
        self.kalman = KalmanBiasCorrector()
        self.ar1 = AR1Corrector()
        self.is_fitted = False

    def fit_residuals(self, df: pd.DataFrame):
        """Fit AR(1) correction on training residuals.
        
        Call this after all components are fitted, passing the full training DataFrame.
        """
        df_prep = df.copy()
        df_prep['Temp'] = df_prep['temperature_c']
        df_prep['Is_Holiday'] = df_prep['is_holiday']
        pred = self.predict(df_prep, apply_ar1=False, apply_kalman=False)
        residuals = df_prep['demand_mw'].values - np.array(pred['forecast_mw'])
        self.ar1.fit(residuals)

    def compute_acceleration(self, df: pd.DataFrame) -> dict:
        """Detect Holt-Winters trend acceleration from recent actuals.
        
        Compares the extrapolated HW trend vs actual recent daily means.
        Returns acceleration factors for forecast adjustment.
        """
        if df is None or df.empty or not hasattr(self.trend, 'daily_smoothed') or not self.trend.daily_smoothed:
            return {'acceleration': 0.0, 'momentum': 0.0, 'short_trend_mw': 0.0, 'medium_trend_mw': 0.0}

        recent = df.tail(14 * 24).copy()
        if len(recent) < 48:
            return {'acceleration': 0.0, 'momentum': 0.0, 'short_trend_mw': 0.0, 'medium_trend_mw': 0.0}

        daily_mean = recent.groupby('Date')['demand_mw'].mean()
        short_t = daily_mean.tail(3).mean() if len(daily_mean) >= 3 else daily_mean.mean()
        medium_t = daily_mean.tail(14).mean() if len(daily_mean) >= 14 else daily_mean.mean()

        # HW trend at last date
        hw_trend = self.trend.last_val
        accel = (short_t / medium_t - 1) if medium_t > 0 else 0.0
        momentum = (short_t / hw_trend - 1) if hw_trend > 0 else 0.0
        return {
            'acceleration': float(accel),
            'momentum': float(momentum),
            'short_trend_mw': float(short_t),
            'medium_trend_mw': float(medium_t),
        }

    def compute_prediction_intervals(self, df: pd.DataFrame, n_hours: int = 24, quantiles: list = None) -> dict:
        """Compute empirical prediction intervals from recent structural residuals."""
        if quantiles is None:
            quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        if df is None or len(df) < 100:
            return {'method': 'no_data', **{q: np.zeros(n_hours) for q in quantiles}}

        # Get structural residuals from recent history
        recent = df.tail(RESIDUAL_HISTORY_HOURS).copy()
        if len(recent) < 100:
            recent = df.copy()

        # Compute structural forecast for recent data
        pred = self.predict(recent, apply_ar1=False, apply_kalman=False)
        residuals = recent['demand_mw'].values - np.array(pred['forecast_mw'])
        residuals = residuals[np.isfinite(residuals)]

        if len(residuals) < 10:
            return {'method': 'insufficient_data', 'n_residuals': 0, **{q: np.zeros(n_hours) for q in quantiles}}

        sorted_res = np.sort(residuals)
        n = len(sorted_res)
        result = {'method': 'empirical_quantile', 'n_residuals': n}
        for q in quantiles:
            idx = int(np.clip(q * n, 0, n - 1))
            result[q] = np.full(n_hours, float(sorted_res[idx]))
        return result

    def save(self, path: str):
        state = {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "temp": self.temp,
            "holiday": self.holiday,
            "growth": self.growth,
            "ar1": self.ar1,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, path)
        logger.info(f"DecomEngineHourly saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"DecomEngineHourly state not found at {path}")
            return
        state = joblib.load(path)
        self.trend = state["trend"]
        self.seasonal = state["seasonal"]
        self.temp = state["temp"]
        self.holiday = state["holiday"]
        self.growth = state.get("growth", GrowthEngine())
        self.ar1 = state.get("ar1", AR1Corrector())
        self.is_fitted = state.get("is_fitted", False)
        logger.info(f"DecomEngineHourly loaded from {path}")

    def predict(self, df: pd.DataFrame, apply_ar1: bool = True, apply_kalman: bool = True) -> Dict[str, Any]:
        t_arr = self.trend.get_trend_array(df['Date'])
        s_arr = self.seasonal.apply(df['Hour'].values, df['DOW'].values)
        tmp_arr = self.temp.apply(df['Temp'].values, df['Hour'].values)
        h_arr = self.holiday.apply(df['Hour'].values, df['Is_Holiday'].values)
        growth_arr = np.array([
            self.growth.get_growth_mult(pd.Timestamp(d).date()) / self.growth.baseline_mult
            for d in df['Date']
        ])

        raw_struct = t_arr * s_arr * tmp_arr * h_arr * growth_arr
        corrected = raw_struct.copy()
        if apply_ar1:
            corrected = self.ar1.correct(corrected)
        if apply_kalman:
            corrected = self.kalman.correct(corrected)

        # Components decomposition based on final structural forecast
        comp_before = raw_struct
        ar1_effect = corrected - raw_struct if apply_ar1 else np.zeros_like(raw_struct)
        kalman_effect = np.full(len(df), self.kalman.bias if apply_kalman else 0.0)

        # Trend acceleration
        accel = self.compute_acceleration(df)

        # Prediction intervals
        pi = self.compute_prediction_intervals(df, n_hours=len(df))
        quantile_fc = {}
        for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
            if q in pi:
                quantile_fc[f'q{int(q*100)}'] = (corrected + pi[q]).tolist()

        return {
            "forecast_mw": corrected.tolist(),
            "quantiles": quantile_fc,
            "prediction_intervals": {
                "method": pi.get('method', 'unknown'),
                "n_residuals": pi.get('n_residuals', 0),
            },
            "components": {
                "trend": (t_arr * s_arr).tolist(),
                "temp_effect": (t_arr * s_arr * (tmp_arr - 1.0)).tolist(),
                "holiday_effect": (t_arr * s_arr * tmp_arr * (h_arr - 1.0)).tolist(),
                "growth_effect": (t_arr * s_arr * tmp_arr * h_arr * (growth_arr - 1.0)).tolist(),
                "ar1_effect": ar1_effect.tolist(),
                "kalman_bias": kalman_effect.tolist(),
            },
            "factors": {
                "trend_mw": t_arr.tolist(),
                "seasonal_ratio": s_arr.tolist(),
                "temp_ratio": tmp_arr.tolist(),
                "holiday_ratio": h_arr.tolist(),
                "growth_ratio": growth_arr.tolist(),
                "acceleration": float(accel['acceleration']),
                "momentum": float(accel['momentum']),
                "short_trend_mw": float(accel['short_trend_mw']),
                "medium_trend_mw": float(accel['medium_trend_mw']),
            }
        }
