import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class FeatureEngine:
    """
    Production Feature Engineering Service.
    Bridge between raw data and model input.
    """
    
    def __init__(self):
        self.target_col = "Community_Load_MW"
        self.nominal_frequency = 50.0
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for the given dataframe.
        
        Args:
            df: DataFrame with timestamps and raw columns
            
        Returns:
            DataFrame with engineered features
        """
        try:
            # operate on a copy
            df_out = df.copy()
            
            # Ensure datetime index
            if not isinstance(df_out.index, pd.DatetimeIndex):
                if 'timestamp' in df_out.columns:
                    df_out['timestamp'] = pd.to_datetime(df_out['timestamp'])
                    df_out.set_index('timestamp', inplace=True)
            
            # 1. Target Variable (if raw components exist)
            if all(col in df_out.columns for col in ["LINE1_MW", "LINE2_MW", "LINE3_MW"]):
                 # Assuming LINE1/2/3 roughly map to T1/T3/T4 in research or just use Total Load
                 # In backend we have "TOTAL_LOAD_MW" which is likely the target
                 if "TOTAL_LOAD_MW" in df_out.columns:
                     df_out[self.target_col] = df_out["TOTAL_LOAD_MW"]
            elif "TOTAL_LOAD_MW" in df_out.columns:
                 df_out[self.target_col] = df_out["TOTAL_LOAD_MW"]
            
            if self.target_col not in df_out.columns:
                logger.warning(f"Target column {self.target_col} could not be created or found")
            
            # 2. Temporal Features
            logger.info(f"Creating temporal features. Columns: {df_out.columns.tolist()}")
            df_out = self._create_temporal_features(df_out)
            
            # 3. Exogenous Features
            logger.info("Creating exogenous features...")
            df_out = self._create_exogenous_features(df_out)
            
            # 4. Lag Features
            logger.info("Creating lag features...")
            df_out = self._create_lag_features(df_out)
            
            # 5. Rolling Features
            df_out = self._create_rolling_features(df_out)
            
            # 6. Clean/Fill NaNs for production safety (optional but recommended)
            # In training we might drop, in inference we might need to fill or handle
            df_out = df_out.ffill().bfill()
            
            return df_out
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise e

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Cyclical encoding
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features based on config."""
        if self.target_col not in df.columns:
            return df
            
        for lag in settings.lag_features_list:
            col_name = f"Lag_{lag}"
            df[col_name] = df[self.target_col].shift(lag)
            
        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features based on config."""
        if self.target_col not in df.columns:
            return df
            
        for window in settings.rolling_windows_list:
            # Research used 'shift(1)' to prevent leakage
            shifted = df[self.target_col].shift(1)
            
            # 24h window (96 steps) specific naming from research
            suffix = "24h" if window == 96 else f"{window}steps"
            
            df[f"Rolling_Mean_{suffix}"] = shifted.rolling(window=window, min_periods=1).mean()
            df[f"Rolling_Std_{suffix}"] = shifted.rolling(window=window, min_periods=1).std()
            df[f"Rolling_Min_{suffix}"] = shifted.rolling(window=window, min_periods=1).min()
            df[f"Rolling_Max_{suffix}"] = shifted.rolling(window=window, min_periods=1).max()
            
        return df

    def _create_exogenous_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create grid-related features."""
        # Using column names from ValidatedData model which map to research
        # ValidatedData has voltage_kv, frequency_hz, etc.
        
        # Map DB columns to research feature names if needed
        # Research: NY6ZA_Flow (from LINE?), Freq_Deviation
        
        # Frequency Deviation
        # Frequency Deviation
        if "frequency_hz" in df.columns:
            logger.info(f"Processing frequency_hz. Dtype: {df['frequency_hz'].dtype}")
            # Ensure numeric
            df["frequency_hz"] = pd.to_numeric(df["frequency_hz"], errors='coerce')
            df["Freq_Deviation"] = (df["frequency_hz"] - self.nominal_frequency).abs()
        elif "FREQ_HZ" in df.columns:
            df["FREQ_HZ"] = pd.to_numeric(df["FREQ_HZ"], errors='coerce')
            df["Freq_Deviation"] = (df["FREQ_HZ"] - self.nominal_frequency).abs()
            
        # NY6ZA / Line 1 Flow (Proxies)
        # If we only have 'line1_mw' etc
        if "line1_mw" in df.columns:
             df["NY6ZA_Flow"] = df["line1_mw"]
             df["NY6ZA_Lag_1"] = df["line1_mw"].shift(1)
             
        # T2 Generation (Negative Load)
        # If specific column exists. In validated data we have line2_mw
        if "line2_mw" in df.columns:
            logger.info(f"Processing line2_mw. Dtype: {df['line2_mw'].dtype}")
            df["line2_mw"] = pd.to_numeric(df["line2_mw"], errors='coerce')
            df["T2_Generation"] = df["line2_mw"].abs()
            
        # Temperature
        if "temperature_c" in df.columns:
            df["Temp_T1_Winding"] = df["temperature_c"] # Proxy since we have one temp
            
        return df
