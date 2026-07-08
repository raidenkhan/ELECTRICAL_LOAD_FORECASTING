"""Intraday corrector for DLinear forecasts.

ARDRegression primary + Lag-1 Dampened online fallback.
Operates in raw MW space — all inputs/outputs are raw MW values.
"""

import pickle
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression
from sklearn.preprocessing import StandardScaler

from app.core.logging import get_logger

logger = get_logger(__name__)

# Full feature set (with lag features). Used in ablation — requires
# sequential simulation where err_lag_{1,2,24} are from TRUE errors.
CORR_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "weekend",
    "err_lag_1", "err_lag_2", "err_lag_24",
    "rolling_err_6h",
    "temperature_c",
]

# Batch-prediction features: NO lag features.
# Lag features cause distribution shift in production (trained on true errors,
# applied to corrected/approximated errors). Only use features that are
# known at batch-prediction time without iterative propagation.
BATCH_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "weekend",
    "temperature_c",
]


class IntradayCorrector:
    """ARDRegression error corrector with iterative batch-prediction support.

    Primary: Pre-trained ARDRegression predicts DLinear error (MW) from
    raw features (MW + cyclical). Handles err_lag_1 by propagating
    approximated errors through 24 forecast hours.

    Online fallback: Lag-1 Dampened (a=0.79) when ARD is unavailable.
    """

    MAX_STALE_DAYS = 7

    def __init__(self, lag1_alpha: float = 0.79, model: Optional[ARDRegression] = None,
                 scaler: Optional[StandardScaler] = None,
                 feature_cols: Optional[List[str]] = None):
        self.lag1_alpha = lag1_alpha
        self.feature_cols = feature_cols or BATCH_FEATURES
        self._model = model
        self._scaler = scaler
        self._is_trained = model is not None

        self._lag1_ema: float = 0.0
        self._lag1_initialized: bool = False
        self._residual_std: float = 50.0  # MW, prior
        self.last_update_time: Optional[datetime] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train ARDRegression on scaled features and raw MW errors."""
        with self._lock:
            self._model = ARDRegression(compute_score=True, tol=1e-4)
            self._model.fit(X, y)
            self._is_trained = True
            preds = self._model.predict(X)
            residuals = np.abs(y - preds)
            self._residual_std = float(np.median(residuals)) + 1e-6
            logger.info(
                f"ARD trained: {X.shape[0]} samples, {X.shape[1]} features, "
                f"residual MAD={self._residual_std:.1f} MW"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(self, raw_pred_mw: np.ndarray,
              features_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Apply correction to raw DLinear prediction (raw MW).

        Uses ARDRegression to predict error from batch-available features
        (cyclical + temperature). No err_lag propagation needed.

        Args:
            raw_pred_mw: (24,) DLinear ensemble prediction in raw MW
            features_df: (24, N) raw features with BATCH_FEATURES columns.
                         If None, falls back to Lag-1 Dampened bias.

        Returns:
            (24,) corrected prediction in raw MW
        """
        with self._lock:
            if features_df is None or len(features_df) < 24:
                return raw_pred_mw.copy() + self._get_damped_bias()

            corrected = np.zeros(24)
            for h in range(24):
                corr = self._predict_hour(features_df.iloc[h])
                corrected[h] = raw_pred_mw[h] + corr

            if self._lag1_initialized:
                last_err = corrected[-1] - raw_pred_mw[-1]
                self._lag1_ema = self.lag1_alpha * last_err + (1 - self.lag1_alpha) * self._lag1_ema
            else:
                self._lag1_ema = 0.0
                self._lag1_initialized = True

            return corrected

    def update(self, prediction_mw: np.ndarray, actual_mw: np.ndarray):
        """Update corrector with observed prediction and actual (raw MW)."""
        with self._lock:
            self.last_update_time = datetime.utcnow()

    def get_error_std(self) -> np.ndarray:
        with self._lock:
            if self._is_stale():
                return np.ones(24) * 50.0
            return np.ones(24) * self._residual_std

    def reset(self):
        with self._lock:
            self._lag1_ema = 0.0
            self._lag1_initialized = False
            self.last_update_time = None

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "type": "intraday_corrector",
                "is_trained": self._is_trained,
                "lag1_ema": float(self._lag1_ema),
                "lag1_initialized": self._lag1_initialized,
                "residual_std_mw": float(self._residual_std),
                "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
                "stale": self._is_stale(),
            }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "lag1_alpha": self.lag1_alpha,
                "feature_cols": self.feature_cols,
                "is_trained": self._is_trained,
                "lag1_ema": float(self._lag1_ema),
                "lag1_initialized": self._lag1_initialized,
                "residual_std": float(self._residual_std),
                "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            }

    def load_state_dict(self, state: Dict[str, Any]):
        with self._lock:
            self.lag1_alpha = state.get("lag1_alpha", 0.79)
            self.feature_cols = state.get("feature_cols", BATCH_FEATURES)
            self._lag1_ema = float(state.get("lag1_ema", 0.0))
            self._lag1_initialized = state.get("lag1_initialized", False)
            self._residual_std = float(state.get("residual_std", 50.0))
            ts = state.get("last_update_time")
            self.last_update_time = datetime.fromisoformat(ts) if ts else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _is_stale(self) -> bool:
        if self.last_update_time is None:
            return True
        return (datetime.utcnow() - self.last_update_time).days >= self.MAX_STALE_DAYS

    def _get_damped_bias(self) -> np.ndarray:
        if self._lag1_initialized:
            return np.ones(24) * self.lag1_alpha * self._lag1_ema
        return np.zeros(24)

    def _predict_hour(self, row: pd.Series) -> float:
        if not self._is_trained or self._model is None:
            if self._lag1_initialized:
                return self.lag1_alpha * self._lag1_ema
            return 0.0

        feats = {}
        for c in self.feature_cols:
            feats[c] = float(row.get(c, 0.0))

        arr = np.array([feats[c] for c in self.feature_cols], dtype=np.float32).reshape(1, -1)
        try:
            if self._scaler is not None:
                arr = self._scaler.transform(arr)
            return float(self._model.predict(arr)[0])
        except Exception as e:
            logger.warning(f"ARD predict failed: {e}")
            return 0.0
