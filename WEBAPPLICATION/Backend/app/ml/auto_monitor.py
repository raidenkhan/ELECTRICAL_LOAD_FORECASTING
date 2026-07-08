"""Automated forecast quality monitor - periodic drift checks via background task.

Runs as an asyncio task during the FastAPI lifespan.
Checks rolling MAE/MAPE against baseline and logs warnings when drift is detected.
Can be extended to trigger automatic retraining.

Usage:
    # In app/main.py lifespan:
    from app.ml.auto_monitor import start_monitor, stop_monitor
    monitor_task = await start_monitor()
    yield
    await stop_monitor(monitor_task)
"""

import asyncio
from datetime import datetime
from typing import Optional

from app.core.logging import get_logger
from app.ml.metrics_service import MetricsService

logger = get_logger(__name__)

CHECK_INTERVAL_SECONDS = 6 * 3600
ALERT_COOLDOWN_HOURS = 24


class AutoMonitor:
    """Background monitor that periodically checks forecast drift."""

    def __init__(self, check_interval: int = CHECK_INTERVAL_SECONDS):
        self.metrics = MetricsService()
        self.check_interval = check_interval
        self._last_alert_time: Optional[datetime] = None
        self._consecutive_drifts = 0
        self._task: Optional[asyncio.Task] = None

    async def _run_cycle(self):
        """One monitoring cycle: check drift, log results, alert if needed."""
        try:
            drift = await self.metrics.check_drift()
            current_mae = drift.get("current_mae")
            degradation_pct = drift.get("degradation_pct", 0.0)
            drift_detected = drift.get("drift_detected", False)
            n_samples = drift.get("count", 0)

            if current_mae is None:
                logger.debug("AutoMonitor: insufficient data for drift check")
                return

            logger.info(
                f"AutoMonitor: MAE={current_mae:.1f} MW, "
                f"degradation={degradation_pct:+.1f}%, "
                f"drift={drift_detected}, n_samples={n_samples}"
            )

            if drift_detected:
                self._consecutive_drifts += 1
                now = datetime.utcnow()
                cooldown_ok = (
                    self._last_alert_time is None
                    or (now - self._last_alert_time).total_seconds() > ALERT_COOLDOWN_HOURS * 3600
                )
                if cooldown_ok:
                    self._last_alert_time = now
                    logger.warning(
                        f"DRIFT DETECTED (x{self._consecutive_drifts}): "
                        f"MAE={current_mae:.1f} MW ({degradation_pct:+.1f}% vs baseline). "
                        f"Consider retraining DLinear."
                    )
                else:
                    logger.debug(f"AutoMonitor: drift persists (x{self._consecutive_drifts}), in cooldown")
            else:
                self._consecutive_drifts = 0

        except Exception as e:
            logger.warning(f"AutoMonitor cycle failed: {e}")

    async def run_forever(self):
        """Run monitoring cycles indefinitely."""
        logger.info(f"AutoMonitor started (interval={self.check_interval}s)")
        try:
            while True:
                await self._run_cycle()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.info("AutoMonitor task cancelled")
        except Exception as e:
            logger.error(f"AutoMonitor crashed: {e}")

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()


_monitor = AutoMonitor()


async def start_monitor(interval: Optional[int] = None) -> asyncio.Task:
    """Start the background monitor task."""
    global _monitor
    if interval is not None:
        _monitor = AutoMonitor(check_interval=interval)
    if _monitor.is_running():
        logger.warning("AutoMonitor already running")
        return _monitor._task
    _monitor._task = asyncio.create_task(_monitor.run_forever())
    return _monitor._task


async def stop_monitor(task: Optional[asyncio.Task] = None):
    """Stop the background monitor task."""
    global _monitor
    t = task or _monitor._task
    if t and not t.done():
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    logger.info("AutoMonitor stopped")

