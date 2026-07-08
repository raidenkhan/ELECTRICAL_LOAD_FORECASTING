"""Drift detection for production forecast monitoring.

CUSUM (Cumulative Sum) and rolling MAE tracker.
Designed to run online: feed it (actual, forecast) pairs, it alarms when drift exceeds threshold.

Usage:
    detector = DriftDetector(threshold_mae_pct=20, min_history=168)
    for actual, forecast in stream:
        detector.update(actual, forecast)
        if detector.drift_alarm:
            logger.warning(f"Drift detected: MAE {detector.rolling_mae:.0f} MW (+{detector.mae_increase_pct:.0f}%)")
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DriftDetector:
    """CUSUM + rolling MAE drift detector for forecast monitoring."""

    # Window config
    short_window: int = 168          # 7 days of hourly data for current performance
    long_window: int = 720           # 30 days of hourly data for baseline
    min_history: int = 168           # minimum data before alarms activate

    # CUSUM config
    cusum_threshold: float = 5.0     # z-score threshold for CUSUM alarm
    cusum_drift: float = 0.5         # allowable drift in z-score units

    # MAE alarm config
    threshold_mae_pct: float = 20.0  # % increase over baseline that triggers alarm
    min_mae_abs: float = 30.0        # minimum absolute MAE increase to avoid noise alarms

    # State
    short_errors: list = field(default_factory=list)
    long_errors: list = field(default_factory=list)
    baseline_mae: Optional[float] = None
    rolling_mae: Optional[float] = None
    mae_increase_pct: float = 0.0
    drift_alarm: bool = False
    cusum_alarm: bool = False
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    n_updates: int = 0

    def update(self, actual: float, forecast: float):
        """Feed one (actual, forecast) pair. Returns self for chaining."""
        if not np.isfinite(actual) or not np.isfinite(forecast):
            return self

        error = actual - forecast
        abs_error = abs(error)

        self.short_errors.append(abs_error)
        self.long_errors.append(abs_error)

        # Keep windows bounded
        if len(self.short_errors) > self.short_window:
            self.short_errors.pop(0)
        if len(self.long_errors) > self.long_window:
            self.long_errors.pop(0)

        self.n_updates += 1

        if self.n_updates < self.min_history:
            return self

        # Compute rolling MAE
        self.rolling_mae = float(np.mean(self.short_errors))

        # Compute baseline (long-window average)
        baseline = float(np.mean(self.long_errors))
        if self.baseline_mae is None:
            self.baseline_mae = baseline

        # MAE drift alarm
        if self.baseline_mae > 0:
            increase = self.rolling_mae - self.baseline_mae
            self.mae_increase_pct = (increase / self.baseline_mae) * 100
            self.drift_alarm = (
                self.mae_increase_pct > self.threshold_mae_pct
                and increase > self.min_mae_abs
            )
        else:
            self.drift_alarm = False

        # CUSUM: track cumulative deviations from target mean (baseline MAE)
        # Use residuals from the baseline as the "target zero"
        target_dev = abs_error - baseline
        self.cusum_pos = max(0.0, self.cusum_pos + target_dev - self.cusum_drift)
        self.cusum_neg = min(0.0, self.cusum_neg + target_dev + self.cusum_drift)

        # Standard deviation of long errors for z-score normalization
        if len(self.long_errors) > 10:
            std_err = float(np.std(self.long_errors)) + 1e-8
            cusum_z_pos = self.cusum_pos / std_err
            cusum_z_neg = abs(self.cusum_neg) / std_err
            self.cusum_alarm = max(cusum_z_pos, cusum_z_neg) > self.cusum_threshold
        else:
            self.cusum_alarm = False

        return self

    def reset(self):
        """Reset all state (e.g., after retraining)."""
        self.short_errors.clear()
        self.long_errors.clear()
        self.baseline_mae = None
        self.rolling_mae = None
        self.mae_increase_pct = 0.0
        self.drift_alarm = False
        self.cusum_alarm = False
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.n_updates = 0

    def get_status(self) -> dict:
        """Return current status as a dict for logging/monitoring."""
        return {
            "rolling_mae_mw": self.rolling_mae,
            "baseline_mae_mw": self.baseline_mae,
            "mae_increase_pct": self.mae_increase_pct,
            "drift_alarm": self.drift_alarm,
            "cusum_alarm": self.cusum_alarm,
            "cusum_z_pos": float(self.cusum_pos),
            "cusum_z_neg": float(abs(self.cusum_neg)),
            "n_updates": self.n_updates,
        }


@dataclass
class ModelMonitor:
    """Production model monitor that evaluates forecasts against actuals.
    
    Tracks per-horizon MAE, drift, and 95th percentile error.
    Runs as singleton updated by the dispatch service when actuals come in.
    
    Usage:
        monitor = ModelMonitor()
        # After new actual data arrives:
        monitor.record_actuals(forecast_date, predicted_values, actual_values)
        if monitor.needs_retraining():
            trigger_retraining()
    """

    horizon_days: int = 1
    max_history: int = 90  # days of history to keep

    # Internal
    daily_maes: list = field(default_factory=list)
    daily_p95: list = field(default_factory=list)
    daily_dates: list = field(default_factory=list)
    drift_detector: DriftDetector = field(default_factory=DriftDetector)
    _last_alarm_date: Optional[str] = None

    def record_actuals(self, forecast_date: str, predicted: np.ndarray, actual: np.ndarray):
        """Record one day of actuals vs forecast."""
        if len(predicted) == 0 or len(actual) == 0:
            return
        mae = float(np.mean(np.abs(actual - predicted)))
        p95 = float(np.percentile(np.abs(actual - predicted), 95))

        self.daily_maes.append(mae)
        self.daily_p95.append(p95)
        self.daily_dates.append(forecast_date)

        # Keep bounded
        if len(self.daily_maes) > self.max_history:
            self.daily_maes.pop(0)
            self.daily_p95.pop(0)
            self.daily_dates.pop(0)

        # Feed drift detector
        for a, p in zip(actual, predicted):
            self.drift_detector.update(float(a), float(p))

        # Log alarm
        if self.drift_detector.drift_alarm:
            logger.warning(
                f"ModelMonitor: Drift detected on {forecast_date}! "
                f"MAE={self.drift_detector.rolling_mae:.0f} MW "
                f"(baseline={self.drift_detector.baseline_mae:.0f} MW, "
                f"+{self.drift_detector.mae_increase_pct:.0f}%)"
            )
            self._last_alarm_date = forecast_date

    def get_recent_mae(self, days: int = 7) -> float:
        """Mean MAE over last N days."""
        if len(self.daily_maes) == 0:
            return 0.0
        recent = self.daily_maes[-min(days, len(self.daily_maes)):]
        return float(np.mean(recent))

    def needs_retraining(self) -> bool:
        """Trigger retraining if drift alarm persisted > 7 days."""
        if not self.drift_detector.drift_alarm and not self.drift_detector.cusum_alarm:
            return False
        # Check if last 7 days consistently above baseline
        if len(self.daily_maes) < 7:
            return False
        recent = self.daily_maes[-7:]
        baseline = float(np.mean(self.daily_maes[:-7])) if len(self.daily_maes) > 14 else 200.0
        return np.mean(recent) > baseline * 1.3

    def get_summary(self) -> dict:
        """Return performance summary."""
        recent_mae = float(np.mean(self.daily_maes[-7:])) if len(self.daily_maes) >= 7 else 0
        overall_mae = float(np.mean(self.daily_maes)) if self.daily_maes else 0
        return {
            "overall_mae_mw": overall_mae,
            "recent_7d_mae_mw": recent_mae,
            "drift_status": self.drift_detector.get_status(),
            "needs_retraining": self.needs_retraining(),
            "last_alarm_date": self._last_alarm_date,
            "days_tracked": len(self.daily_maes),
        }
