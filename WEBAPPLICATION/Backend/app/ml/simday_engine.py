import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import datetime
from app.core.logging import get_logger

logger = get_logger(__name__)

class SimDayEngine:
    """
    Similar-Day Method — Automated GRIDCo-Excel analogy.
    Finds K nearest historical days by feature distance (DOW, Month, Temp, etc.)
    and calculates a weighted average of their actual load profiles.
    """
    
    def __init__(self, k_similar_days: int = 5):
        self.k = k_similar_days
        self.steps_per_day = 96
        self.sim_feat_cols = ['DayOfWeek', 'Month', 'Is_Weekend', 'Is_Holiday', 
                             'Mean_Temp', 'Prev_Mean', 'Roll7_Mean']
        self.train_profiles = {}
        self.train_day_feats = None
        self.col_min = None
        self.col_rng = None
        self.train_norm = None
        self.is_fitted = False

    def fit(self, df_history: pd.DataFrame, holidays: set = None):
        """
        Builds the historical lookup and pre-calculates feature normalization.
        """
        try:
            logger.info("Fitting SimDayEngine on historical data...")
            
            # Ensure dataframe has required columns
            df = df_history.copy()
            if 'Date' not in df.columns:
                df['Date'] = df.index.date
            if 'DayOfWeek' not in df.columns:
                df['DayOfWeek'] = df.index.dayofweek
            if 'Month' not in df.columns:
                df['Month'] = df.index.month
            if 'TimeSlot' not in df.columns:
                df['TimeSlot'] = df.index.hour * 4 + df.index.minute // 15
            if 'Is_Weekend' not in df.columns:
                df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
            if 'Is_Holiday' not in df.columns:
                if holidays:
                    df['Is_Holiday'] = df.index.strftime('%Y-%m-%d').isin(holidays).astype(int)
                else:
                    df['Is_Holiday'] = 0
            if 'Temperature' not in df.columns and 'temperature_c' in df.columns:
                df['Temperature'] = df['temperature_c']
            elif 'Temperature' not in df.columns:
                df['Temperature'] = 28.0

            profiles = {}
            records = []

            # Precompute daily means for all dates
            daily_means = df.groupby('Date')['total_load_mw'].mean().sort_index()
            dates_list = list(daily_means.index)

            for date, grp in df.groupby('Date'):
                grp = grp.sort_values('TimeSlot')
                if len(grp) < self.steps_per_day:
                    continue
                profiles[date] = grp['total_load_mw'].values[:self.steps_per_day]

                idx = dates_list.index(date)
                prev_mean = float(daily_means.iloc[idx - 1]) if idx > 0 else float(daily_means.iloc[0])
                roll7_mean = float(daily_means.iloc[max(0, idx - 7):idx].mean()) if idx > 0 else prev_mean

                records.append({
                    'Date': date,
                    'DayOfWeek': grp['DayOfWeek'].iloc[0],
                    'Month': grp['Month'].iloc[0],
                    'Is_Weekend': grp['Is_Weekend'].iloc[0],
                    'Is_Holiday': grp['Is_Holiday'].iloc[0],
                    'Mean_Temp': grp['Temperature'].mean(),
                    'Prev_Mean': prev_mean,
                    'Roll7_Mean': roll7_mean,
                })

            if not records:
                logger.warning("No complete days found in history for SimDay fit.")
                return

            self.train_day_feats = pd.DataFrame(records).set_index('Date')
            self.train_profiles = profiles
            
            valid_train_dates = [d for d in self.train_day_feats.index if d in self.train_profiles]
            train_mat = self.train_day_feats.loc[valid_train_dates, self.sim_feat_cols].values.astype(float)

            # Normalize by training set statistics
            self.col_min = train_mat.min(axis=0)
            self.col_rng = train_mat.max(axis=0) - self.col_min + 1e-8
            self.train_norm = (train_mat - self.col_min) / self.col_rng
            self.valid_train_dates = valid_train_dates
            
            self.is_fitted = True
            logger.info(f"SimDayEngine fitted successfully with {len(valid_train_dates)} days.")
            
        except Exception as e:
            logger.error(f"Failed to fit SimDayEngine: {str(e)}")
            self.is_fitted = False

    def predict(self, test_date: datetime.date, target_features: Dict[str, Any]) -> np.ndarray:
        """
        Generates a 96-step forecast for a specific day using the GRIDCo "Like-for-Like" method.
        Includes:
        1. Temperature sensitivity adjustment (approx 3% per degree Celsius)
        2. Annual demand growth scaling (8% per year)
        3. Recency-weighted distance
        """
        if not self.is_fitted:
            logger.warning("SimDayEngine not fitted. Returning flat persistence.")
            return np.full(self.steps_per_day, 120.0)

        try:
            # 1. Feature Vector Construction
            test_feat = np.array([target_features[c] for c in self.sim_feat_cols]).astype(float)
            test_norm = (test_feat - self.col_min) / self.col_rng

            # 2. Selection: Euclidean distances to all training days
            dists = np.sqrt(((self.train_norm - test_norm) ** 2).sum(axis=1))

            # 3. Find K Nearest Profiles
            k_idx = np.argsort(dists)[:self.k]
            k_dates = [self.valid_train_dates[i] for i in k_idx]
            
            # Inverse distance weighting
            k_weights = 1.0 / (dists[k_idx] + 1e-6)
            k_weights /= k_weights.sum()

            adjusted_profiles = []
            
            # GRIDCo interview constants
            TEMP_COEFFICIENT = 0.031  # ~3.1% change per degree Celsius
            ANNUAL_GROWTH = 0.08      # 8% annual growth in demand

            for i, date in enumerate(k_dates):
                profile = self.train_profiles[date].copy()
                hist_feats = self.train_day_feats.loc[date]
                
                # A. Temperature Adjustment (GRIDCo Ratio)
                # If target is hotter than historical, scale profile UP
                temp_delta = target_features['Mean_Temp'] - hist_feats['Mean_Temp']
                temp_mult = 1.0 + (temp_delta * TEMP_COEFFICIENT)
                
                # B. Demand Growth Adjustment (Year-over-Year)
                # Scale old data up to the target date's regime
                days_diff = (test_date - date).days
                years_diff = days_diff / 365.25
                growth_mult = 1.0 + (years_diff * ANNUAL_GROWTH)
                
                # C. Combine Multipliers
                total_mult = max(0.5, min(temp_mult * growth_mult, 2.0)) # Safety clip
                adjusted_profiles.append(profile * total_mult)

            # 4. Weighted Average of Adjusted Profiles
            day_pred = np.average(np.stack(adjusted_profiles), weights=k_weights, axis=0)
            
            return day_pred
            
        except Exception as e:
            logger.error(f"SimDay prediction failed: {str(e)}")
            return np.full(self.steps_per_day, 120.0)
