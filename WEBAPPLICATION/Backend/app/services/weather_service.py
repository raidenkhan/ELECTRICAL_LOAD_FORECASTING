import httpx
import pandas as pd
from typing import Dict, List, Optional, Any
from app.core.logging import get_logger
import time

logger = get_logger(__name__)

class WeatherService:
    """
    Service to fetch real-time and forecast weather data from Open-Meteo.
    Target Coordinates: Nayagina Substation area (approx 10.86, -1.04)
    Reflecting Northern Region, Ghana.
    """
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        # Nayagina Substation, Northern Region
        self.lat = 10.86
        self.lon = -1.04
        self._cache = {"time": 0, "data": None}
        self.cache_ttl = 3600  # 1 hour

    async def get_forecast(self, horizon_hours: int = 24) -> pd.DataFrame:
        """
        Fetches hourly temperature and precipitation forecast.
        """
        if time.time() - self._cache["time"] < self.cache_ttl and self._cache["data"] is not None:
            logger.debug("Serving weather from cache")
            return self._cache["data"]

        try:
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m"],
                "forecast_days": 3  # Sufficient for STLF
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(self.base_url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
                hourly = data.get("hourly", {})
                times = hourly.get("time", [])
                temps = hourly.get("temperature_2m", [])
                precip = hourly.get("precipitation", [])
                rh = hourly.get("relative_humidity_2m", [])
                
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime(times),
                    "temp_c": temps,
                    "precip_mm": precip,
                    "rh_pct": rh
                }).set_index("timestamp")
                
                # Upsample to 15-min to match SCADA resolution
                df_resampled = df.resample("15min").interpolate(method="linear")
                
                self._cache = {"time": time.time(), "data": df_resampled}
                return df_resampled
                
        except Exception as e:
            logger.error(f"Failed to fetch weather from Open-Meteo: {e}")
            # Return empty DF on failure
            return pd.DataFrame()

    async def get_current_temp(self) -> float:
        """Helper to get current estimated temperature."""
        df = await self.get_forecast()
        if df.empty:
            return 28.0
            
        now = pd.Timestamp.now().floor("15min")
        if now in df.index:
            return float(df.loc[now, "temp_c"])
        return 28.0
