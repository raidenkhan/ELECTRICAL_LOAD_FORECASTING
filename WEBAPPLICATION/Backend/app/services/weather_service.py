import httpx
import pandas as pd
from app.core.logging import get_logger
import time
from datetime import datetime
from typing import Optional, List

logger = get_logger(__name__)

ACCRA_MONTHLY_TEMP = {
    1: 27.4, 2: 28.2, 3: 28.4, 4: 28.2,
    5: 27.6, 6: 26.4, 7: 25.6, 8: 25.4,
    9: 26.0, 10: 26.8, 11: 27.4, 12: 27.6,
}

class WeatherService:
    """
    Fetches hourly temperature forecast from Open-Meteo for Accra, Ghana.
    Falls back to Accra seasonal monthly averages if the API is unavailable.
    Supports manual temperature override via admin endpoint.
    """

    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.lat = 5.6037
        self.lon = -0.1870
        self._cache = {"time": 0, "data": None}
        self.cache_ttl = 3600
        self._manual_override: Optional[List[float]] = None

    def set_manual_override(self, temps_c: List[float]):
        self._manual_override = temps_c

    def clear_manual_override(self):
        self._manual_override = None

    def seasonal_fallback(self, start_time: datetime, horizon_hours: int) -> pd.DataFrame:
        month = start_time.month
        base_temp = ACCRA_MONTHLY_TEMP.get(month, 27.0)
        diurnal = [2.5, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -3.5,
                   -3.0, -2.0, 0.0, 1.5, 2.5, 3.0, 3.5, 3.5,
                   3.0, 2.5, 2.0, 1.0, 0.0, -1.0, -2.0, -2.5]
        start_hour = start_time.hour
        temps = []
        for h in range(horizon_hours):
            diurnal_idx = (start_hour + h) % 24
            temps.append(round(base_temp + diurnal[diurnal_idx], 1))
        ts = pd.date_range(start=start_time.replace(minute=0, second=0, microsecond=0),
                           periods=horizon_hours, freq='h')
        return pd.DataFrame({"temp_c": temps}, index=ts)

    async def get_forecast(self, horizon_hours: int = 24) -> pd.DataFrame:
        return await self.get_forecast_from(datetime.now(), horizon_hours)

    async def get_forecast_from(self, start_time: datetime, horizon_hours: int) -> pd.DataFrame:
        result = self.seasonal_fallback(start_time, horizon_hours)

        if self._manual_override is not None:
            logger.info("Serving manual temperature override")
            override_len = min(len(self._manual_override), horizon_hours)
            for i in range(override_len):
                result.iloc[i] = self._manual_override[i]
            return result

        now = datetime.now()
        cache_hit = (time.time() - self._cache["time"] < self.cache_ttl and self._cache["data"] is not None)
        api_df = self._cache["data"] if cache_hit else None

        if not cache_hit:
            try:
                params = {
                    "latitude": self.lat,
                    "longitude": self.lon,
                    "hourly": ["temperature_2m"],
                    "forecast_days": 3,
                    "timezone": "Africa/Accra",
                }
                async with httpx.AsyncClient() as client:
                    response = await client.get(self.base_url, params=params, timeout=10.0)
                    response.raise_for_status()
                    data = response.json()
                    hourly = data.get("hourly", {})
                    times = hourly.get("time", [])
                    temps = hourly.get("temperature_2m", [])
                    if times and temps:
                        api_df = pd.DataFrame({
                            "timestamp": pd.to_datetime(times),
                            "temp_c": temps,
                        }).set_index("timestamp")
                        self._cache = {"time": time.time(), "data": api_df}
                        logger.info(f"Fetched {len(temps)} hourly temps from Open-Meteo for Accra")
            except Exception as e:
                logger.warning(f"Open-Meteo failed: {e}")

        if api_df is not None:
            for i in range(min(len(api_df), horizon_hours)):
                ts = result.index[i]
                if ts in api_df.index:
                    result.iloc[i] = api_df.loc[ts, "temp_c"]

        return result

    async def get_current_temp(self) -> float:
        df = await self.get_forecast()
        if df.empty:
            return ACCRA_MONTHLY_TEMP.get(datetime.now().month, 27.0)
        now = pd.Timestamp.now().floor("h")
        if now in df.index:
            return float(df.loc[now, "temp_c"])
        return ACCRA_MONTHLY_TEMP.get(datetime.now().month, 27.0)
