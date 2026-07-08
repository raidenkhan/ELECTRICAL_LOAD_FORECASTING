"""DLinear + ARD Intraday Correction — production forecast engine.

6-fold ensemble with online error correction, circuit breaker, persistent state.
"""
import json, os, pickle, sqlite3, threading, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from app.core.logging import get_logger
from app.ml.statistical_fallback import StatisticalFallback
from app.ml.intraday_corrector import IntradayCorrector, CORR_FEATURES

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DLinear model (matches the stress-test implementation exactly)
# ---------------------------------------------------------------------------
class _DLinear(nn.Module):
    def __init__(self, n_features, forecast_horizon, input_window, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.trend_linear = nn.Linear(input_window, forecast_horizon)
        self.seasonal_linear = nn.Linear(input_window, forecast_horizon)
        n_cal = n_features - 1
        self.calendar_linear = nn.Linear(input_window * n_cal, forecast_horizon)

    def moving_avg(self, x):
        pad = self.kernel - 1
        xp = nn.functional.pad(x.unsqueeze(1), (pad, 0), mode='replicate').squeeze(1)
        k = torch.ones(self.kernel, device=x.device) / self.kernel
        return nn.functional.conv1d(xp.unsqueeze(1), k.view(1, 1, -1), padding=0).squeeze(1)

    def forward(self, x):
        demand = x[:, :, 0]
        calendar = x[:, :, 1:]
        trend = self.moving_avg(demand)
        seasonal = demand - trend
        B, S, C = calendar.shape
        return (self.trend_linear(trend) + self.seasonal_linear(seasonal)
                + self.calendar_linear(calendar.reshape(B, S * C)))


# ---------------------------------------------------------------------------
# TIDE Bias Corrector (thread-safe) — DEPRECATED
# Replaced by IntradayCorrector. Kept for backward compat with archived runs.
# ---------------------------------------------------------------------------
class _TideCorrector:
    MAX_STALE_DAYS = 7

    def __init__(self, alpha: float = 0.3, window_hours: int = 48):
        self.alpha = alpha
        self.window_hours = window_hours
        self.error_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self._ema_bias: Optional[np.ndarray] = None
        self._ema_error_std: Optional[np.ndarray] = None
        self.last_update_time: Optional[datetime] = None
        self._lock = threading.Lock()

    def _is_stale(self) -> bool:
        if self.last_update_time is None or len(self.error_buffer) == 0:
            return True
        return (datetime.utcnow() - self.last_update_time).days >= self.MAX_STALE_DAYS

    def get_bias(self) -> np.ndarray:
        with self._lock:
            if self._is_stale():
                return np.zeros(24)
            recent = self.error_buffer[-self.window_hours // 24:]
            if len(recent) == 0:
                return np.zeros(24)
            errors = np.array([a - p for p, a in recent])
            bias = np.mean(errors, axis=0)
            if self._ema_bias is not None:
                self._ema_bias = self.alpha * bias + (1 - self.alpha) * self._ema_bias
            else:
                self._ema_bias = bias
            return self._ema_bias.copy()

    def get_error_std(self) -> np.ndarray:
        with self._lock:
            if self._is_stale():
                return np.ones(24) * 0.05
            recent = self.error_buffer[-self.window_hours // 24:]
            if len(recent) == 0:
                return np.ones(24) * 0.05
            errors = np.array([a - p for p, a in recent])
            err_std = np.std(errors, axis=0)
            if self._ema_error_std is not None:
                self._ema_error_std = self.alpha * err_std + (1 - self.alpha) * self._ema_error_std
            else:
                self._ema_error_std = err_std
            return np.maximum(self._ema_error_std.copy(), 1e-6)

    def apply(self, raw_pred: np.ndarray) -> np.ndarray:
        return raw_pred + self.get_bias()

    def update(self, prediction: np.ndarray, actual: np.ndarray):
        with self._lock:
            self.error_buffer.append((prediction.copy(), actual.copy()))
            self.last_update_time = datetime.utcnow()

    def reset(self):
        with self._lock:
            self.error_buffer = []
            self._ema_bias = None
            self._ema_error_std = None
            self.last_update_time = None

    def state_dict(self):
        with self._lock:
            return {
                "alpha": self.alpha,
                "window_hours": self.window_hours,
                "error_buffer": [(p.tolist(), a.tolist()) for p, a in self.error_buffer],
                "ema_bias": self._ema_bias.tolist() if self._ema_bias is not None else None,
                "ema_error_std": self._ema_error_std.tolist() if self._ema_error_std is not None else None,
                "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            }

    def load_state_dict(self, state: dict):
        with self._lock:
            # Persisted alpha must NOT override the constructor's alpha
            self.window_hours = state.get("window_hours", 48)
            self.error_buffer = [(np.array(p), np.array(a)) for p, a in state.get("error_buffer", [])]
            ema = state.get("ema_bias")
            self._ema_bias = np.array(ema) if ema is not None else None
            ema_std = state.get("ema_error_std")
            self._ema_error_std = np.array(ema_std) if ema_std is not None else None
            ts = state.get("last_update_time")
            self.last_update_time = datetime.fromisoformat(ts) if ts else None


# ---------------------------------------------------------------------------
# Feature engineering (mirrors research data_loader.py)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    'demand_mw', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'temperature_c',
]

def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df['date']) if 'date' in df.columns else pd.to_datetime(df['DATETIME'])
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * ts.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * ts.dt.month / 12)
    return df


# ---------------------------------------------------------------------------
# DLinearEngine — production entry point
# ---------------------------------------------------------------------------
class DLinearEngine:
    """6-fold DLinear ensemble + TIDE adaptive level correction.

    Usage:
        engine = DLinearEngine()
        result = engine.predict(history_df, horizon_hours=24)
        # later, when actuals arrive:
        engine.update(actual_mw, result["forecast_mw"])
    """

    INPUT_WINDOW = 168  # 7 days
    FORECAST_HORIZON = 24
    N_FEATURES = 8

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        stats_path: Optional[str] = None,
        db_path: Optional[str] = None,
        corrector_path: Optional[str] = None,
    ):
        base = Path(__file__).parent.parent.parent
        self.checkpoint_dir = Path(checkpoint_dir or base / "models" / "dlinear")
        self.stats_path = Path(stats_path or self.checkpoint_dir / "normalization_stats.json")
        self.db_path = db_path or str(base / "models" / "dlinear" / "corrector_state.db")
        self.corrector_path = Path(corrector_path or self.checkpoint_dir / "intraday_corrector.pkl")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models: List[_DLinear] = []
        self.normalization_stats: dict = {}
        self.corrector = IntradayCorrector()
        self.fallback = StatisticalFallback()

        self.is_fitted = False
        self.last_inference_time: Optional[datetime] = None
        self.inference_count = 0
        self.rolling_mae: List[float] = []
        self._load_models()

    def _load_models(self):
        if not self.checkpoint_dir.exists():
            logger.warning(f"Checkpoint dir not found: {self.checkpoint_dir}")
            return

        if not self.stats_path.exists():
            logger.warning(f"Stats file not found: {self.stats_path}")
            return

        with open(self.stats_path) as f:
            self.normalization_stats = json.load(f)

        ckpts = sorted(self.checkpoint_dir.glob("h10_Fold_*.pt"))
        if not ckpts:
            logger.warning(f"No checkpoints found in {self.checkpoint_dir}")
            return

        loaded = 0
        for ckpt_path in ckpts:
            try:
                model = _DLinear(self.N_FEATURES, self.FORECAST_HORIZON, self.INPUT_WINDOW).to(self.device)
                state = torch.load(ckpt_path, map_location=self.device)
                sd = state.get('model_state_dict', state)
                model.load_state_dict(sd)
                model.eval()
                self.models.append(model)
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load {ckpt_path.name}: {e}")

        logger.info(f"Loaded {loaded}/{len(ckpts)} DLinear models")
        self.is_fitted = loaded > 0

        # Load pre-trained ARD corrector
        self._load_corrector_model()

        # Restore online state from DB
        self._load_corrector_state()

        # Migrate old TIDE state
        self._migrate_old_tide_state()

        # Warm-up inference
        if self.is_fitted:
            self._warmup()

    def _warmup(self):
        try:
            dummy = torch.randn(1, self.INPUT_WINDOW, self.N_FEATURES, device=self.device)
            with torch.no_grad():
                _ = self.models[0](dummy)
            logger.info("DLinear warm-up complete")
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def predict(self, history_df: pd.DataFrame, horizon_hours: int = 24,
                future_temps_c: Optional[List[float]] = None,
                use_tide: bool = True) -> Dict[str, Any]:
        if not self.is_fitted:
            logger.warning("DLinear not fitted — falling back to WeightedTrend")
            return self._fallback_predict(history_df, horizon_hours)

        try:
            return self._predict_internal(history_df, horizon_hours, future_temps_c, use_tide)
        except Exception as e:
            logger.error(f"DLinear predict failed: {e} — falling back to WeightedTrend")
            return self._fallback_predict(history_df, horizon_hours)

    def update(self, actual_mw: np.ndarray, predicted_mw: np.ndarray):
        err = np.mean(np.abs(actual_mw - predicted_mw))
        self.rolling_mae.append(err)
        if len(self.rolling_mae) > 168:
            self.rolling_mae.pop(0)
        # Pass raw MW to the corrector
        self.corrector.update(
            np.array(predicted_mw, dtype=np.float32),
            np.array(actual_mw, dtype=np.float32),
        )
        self._save_corrector_state()

    def health(self) -> Dict[str, Any]:
        corr_health = self.corrector.health()
        return {
            "engine": "dlinear_intraday",
            "checkpoints_loaded": len(self.models),
            "is_fitted": self.is_fitted,
            "last_inference": self.last_inference_time.isoformat() if self.last_inference_time else None,
            "inference_count": self.inference_count,
            "mae_24h": float(np.mean(self.rolling_mae[-24:])) if self.rolling_mae else None,
            **corr_health,
        }

    def reset_bias(self):
        self.corrector.reset()
        self._save_corrector_state()
        logger.info("Corrector state reset")

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------
    def _predict_internal(self, history_df: pd.DataFrame, horizon_hours: int,
                          future_temps_c: Optional[List[float]],
                          use_correction: bool = True) -> Dict[str, Any]:
        if len(history_df) < self.INPUT_WINDOW + self.FORECAST_HORIZON:
            logger.warning(f"History too short ({len(history_df)} rows), need > {self.INPUT_WINDOW + self.FORECAST_HORIZON}")
            return self._fallback_predict(history_df, horizon_hours)

        t0 = time.time()

        last_ts = pd.to_datetime(history_df['date'].iloc[-1]) if 'date' in history_df.columns else pd.to_datetime(history_df['DATETIME'].iloc[-1])
        future_rows = []
        for h in range(horizon_hours):
            ts = last_ts + timedelta(hours=h + 1)
            temp = future_temps_c[h] if future_temps_c and h < len(future_temps_c) else 28.0
            future_rows.append({"date": ts, "demand_mw": 0.0, "temperature_c": temp})

        future_df = pd.DataFrame(future_rows)
        full_df = pd.concat([history_df[['date', 'demand_mw', 'temperature_c']].iloc[-(self.INPUT_WINDOW):], future_df], ignore_index=True)
        full_df = _add_cyclical_features(full_df)
        full_df['weekend'] = full_df['date'].dt.dayofweek.isin([5, 6]).astype(float)

        fold_key = list(self.normalization_stats.keys())[-1]
        stats = self.normalization_stats[fold_key]
        means, stds = stats["means"], stats["stds"]

        norm = full_df.copy()
        for c in FEATURE_COLS:
            norm[c] = (norm[c].values.astype(np.float32) - means[c]) / stds[c]

        features = norm[FEATURE_COLS].values.astype(np.float32)

        X = torch.tensor(features[:self.INPUT_WINDOW], dtype=torch.float32).unsqueeze(0).to(self.device)

        all_preds = []
        with torch.no_grad():
            for model in self.models:
                pred = model(X).detach().cpu().numpy()[0]
                all_preds.append(pred)
        all_preds = np.array(all_preds)
        raw_ensemble_z = np.mean(all_preds, axis=0)
        ensemble_std_per_hour = np.std(all_preds, axis=0)

        demand_std = stds['demand_mw']
        demand_mean = means['demand_mw']
        raw_ensemble_mw = raw_ensemble_z * demand_std + demand_mean

        if use_correction:
            # Build raw features for corrector (24 forecast hours)
            corr_df = full_df.iloc[-horizon_hours:][
                ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                 'month_sin', 'month_cos', 'temperature_c', 'weekend']
            ].reset_index(drop=True)
            corrected_mw = self.corrector.apply(raw_ensemble_mw, corr_df)
        else:
            corrected_mw = raw_ensemble_mw.copy()

        corrected_z = (corrected_mw - demand_mean) / demand_std

        forecast_mw = corrected_mw.tolist()
        raw_mw = raw_ensemble_mw.tolist()

        error_std_mw = self.corrector.get_error_std() if use_correction else np.ones(24) * 50.0
        error_std_z = error_std_mw / demand_std
        ens_var_z = ensemble_std_per_hour ** 2
        uncertainty_z = np.sqrt(error_std_z ** 2 + ens_var_z)
        uncertainty_mw = (uncertainty_z * demand_std).tolist()
        p10_mw = [(f - 1.28 * u) for f, u in zip(forecast_mw, uncertainty_mw)]
        p90_mw = [(f + 1.28 * u) for f, u in zip(forecast_mw, uncertainty_mw)]
        p10_mw = [max(0.0, v) for v in p10_mw]

        bias_correction_mw = (corrected_mw - raw_ensemble_mw).tolist()

        elapsed = time.time() - t0
        self.last_inference_time = datetime.now()
        self.inference_count += 1

        return {
            "engine": "dlinear_intraday",
            "horizon_hours": horizon_hours,
            "forecast_mw": forecast_mw,
            "forecast_raw_mw": raw_mw,
            "ensemble_std_per_hour": ensemble_std_per_hour.tolist(),
            "uncertainty_mw": uncertainty_mw,
            "p10_mw": p10_mw,
            "p90_mw": p90_mw,
            "bias_correction_mw": bias_correction_mw,
            "inference_ms": round(elapsed * 1000, 1),
            "checkpoints_used": len(self.models),
        }

    def _fallback_predict(self, history_df: pd.DataFrame, horizon_hours: int) -> Dict[str, Any]:
        try:
            if 'date' in history_df.columns:
                hist = history_df.rename(columns={'demand_mw': 'total_load_mw'}).set_index('date')
            else:
                hist = history_df.copy()
            if 'total_load_mw' not in hist.columns and 'demand_mw' in hist.columns:
                hist = hist.rename(columns={'demand_mw': 'total_load_mw'})
            hist.index = pd.to_datetime(hist.index)
            self.fallback.fit(hist)
            future = pd.date_range(start=hist.index[-1] + pd.Timedelta(hours=1), periods=horizon_hours, freq='h')
            df_future = pd.DataFrame(index=future)
            pred = self.fallback.predict(df_future)
            return {
                "engine": "statistical_fallback",
                "horizon_hours": horizon_hours,
                "forecast_mw": list(pred),
                "inference_ms": 0,
            }
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")
            return {
                "engine": "error",
                "horizon_hours": horizon_hours,
                "forecast_mw": [0.0] * horizon_hours,
                "error": str(e),
            }

    # -----------------------------------------------------------------------
    # Corrector state persistence via SQLite
    # -----------------------------------------------------------------------
    def _get_db_conn(self, path=None):
        p = path or self.db_path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        conn = sqlite3.connect(p, check_same_thread=False)
        conn.execute("CREATE TABLE IF NOT EXISTS corrector_state (key TEXT PRIMARY KEY, value TEXT)")
        return conn

    def _load_corrector_model(self):
        try:
            if self.corrector_path.exists():
                with open(self.corrector_path, "rb") as f:
                    data = pickle.load(f)
                self.corrector._model = data.get("model")
                self.corrector._scaler = data.get("scaler")
                self.corrector._is_trained = self.corrector._model is not None
                rs = data.get("residual_std")
                if rs is not None:
                    self.corrector._residual_std = float(rs)
                fc = data.get("feature_cols")
                if fc is not None:
                    self.corrector.feature_cols = fc
                logger.info(f"ARD model loaded: {os.path.getsize(self.corrector_path) / 1024:.1f} KB")
        except Exception as e:
            logger.warning(f"Failed to load corrector model: {e}")

    def _migrate_old_tide_state(self):
        old_dir = Path(self.db_path).parent
        for name in ["tide_state.db", "h10_state.db"]:
            old_path = old_dir / name
            if old_path.exists():
                try:
                    os.remove(old_path)
                    logger.info(f"Removed old state db: {name}")
                except Exception as e:
                    logger.warning(f"Removing {name} failed: {e}")

    def _save_corrector_state(self):
        try:
            conn = self._get_db_conn()
            state = self.corrector.state_dict()
            conn.execute("INSERT OR REPLACE INTO corrector_state (key, value) VALUES (?, ?)",
                         ("corrector", json.dumps(state)))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to persist corrector state: {e}")

    def _load_corrector_state(self):
        try:
            conn = self._get_db_conn()
            row = conn.execute("SELECT value FROM corrector_state WHERE key='corrector'").fetchone()
            conn.close()
            if row:
                self.corrector.load_state_dict(json.loads(row[0]))
                logger.info(f"Corrector state restored")
        except Exception as e:
            logger.warning(f"Failed to restore corrector state: {e}")
