import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
from app.core.logging import get_logger

logger = get_logger(__name__)


class StatisticalFallback:
    """
    Simple statistical fallback model using historical load patterns.
    Computes hourly/daily means from data instead of hardcoded values.
    """
    
    def __init__(self, df_hist: Optional[pd.DataFrame] = None):
        self.hourly_means: Dict[int, float] = {}
        self.daily_means: Dict[int, float] = {}
        self.weekly_pattern: Dict[int, float] = {}
        self.overall_mean: float = 125.0  # Default
        self.is_fitted: bool = False
        
        if df_hist is not None:
            self.fit(df_hist)
    
    def fit(self, df_hist: pd.DataFrame) -> "StatisticalFallback":
        """
        Compute statistics from historical data.
        """
        if df_hist.empty:
            logger.warning("Empty dataframe, using default fallback")
            return self
        
        if "total_load_mw" not in df_hist.columns:
            logger.warning("No total_load_mw column, using default fallback")
            return self
        
        loads = df_hist["total_load_mw"].dropna()
        if loads.empty:
            return self
        
        self.overall_mean = float(loads.mean())
        
        # Hourly patterns
        df_with_load = df_hist.copy()
        for hour in range(24):
            mask = df_with_load.index.hour == hour
            hour_data = df_with_load.loc[mask, "total_load_mw"].dropna()
            if not hour_data.empty:
                self.hourly_means[hour] = float(hour_data.mean())
            else:
                self.hourly_means[hour] = self.overall_mean
        
        # Daily pattern (day of week)
        for dow in range(7):
            mask = df_with_load.index.dayofweek == dow
            dow_data = df_with_load.loc[mask, "total_load_mw"].dropna()
            if not dow_data.empty:
                self.daily_means[dow] = float(dow_data.mean())
            else:
                self.daily_means[dow] = self.overall_mean
        
        # Weekly pattern (hour + day of week combined)
        for hour in range(24):
            for dow in range(7):
                mask = (df_with_load.index.hour == hour) & (df_with_load.index.dayofweek == dow)
                combined_data = df_with_load.loc[mask, "total_load_mw"].dropna()
                if not combined_data.empty:
                    self.weekly_pattern[hour * 7 + dow] = float(combined_data.mean())
        
        self.is_fitted = True
        logger.info(f"Fitted statistical fallback: mean={self.overall_mean:.1f}MW, "
                  f"hourly_slots={len(self.hourly_means)}, weekly_slots={len(self.weekly_pattern)}")
        return self
    
    def predict(self, df_future: pd.DataFrame) -> List[float]:
        """
        Generate forecast using learned patterns.
        """
        if not self.is_fitted:
            return [self.overall_mean] * len(df_future)
        
        forecasts = []
        for idx in df_future.index:
            hour = idx.hour
            dow = idx.dayofweek
            
            key = hour * 7 + dow
            if key in self.weekly_pattern:
                val = self.weekly_pattern[key]
            elif hour in self.hourly_means:
                val = self.hourly_means[hour]
            else:
                val = self.overall_mean
            
            forecasts.append(val)
        
        return forecasts


def get_statistical_fallback(
    df_hist: pd.DataFrame,
    horizon_hours: int
) -> Dict[str, Any]:
    """
    Compute a statistical fallback forecast.
    """
    sf = StatisticalFallback(df_hist)
    future_dates = pd.date_range(
        start=df_hist.index[-1] + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="h"
    )
    df_future = pd.DataFrame(index=future_dates)
    
    forecasts = sf.predict(df_future)
    
    return {
        "model_type": "statistical_fallback",
        "base_value": sf.overall_mean,
        "timestamps": future_dates,
        "forecast_mw": forecasts,
        "p10": [v * 0.92 for v in forecasts],
        "p90": [v * 1.08 for v in forecasts],
        "metadata": {
            "n_historical_records": len(df_hist),
            "hourly_pattern": len(sf.hourly_means),
            "is_fitted": sf.is_fitted
        }
    }