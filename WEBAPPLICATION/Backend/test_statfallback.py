import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.ml.statistical_fallback import get_statistical_fallback

# Create sample historical data
n = 168  # 1 week of hourly data
dates = pd.date_range(start=datetime.now() - timedelta(hours=n), periods=n, freq="h")
np.random.seed(42)
loads = 125 + 20 * np.sin(2 * np.pi * dates.hour / 24) + np.random.randn(n) * 5

df_hist = pd.DataFrame({"total_load_mw": loads}, index=dates)

# Test statistical fallback
result = get_statistical_fallback(df_hist, horizon_hours=24)

print(f"Model Type: {result['model_type']}")
print(f"Base Value: {result['base_value']:.2f} MW")
print(f"Forecast (first 3): {result['forecast_mw'][:3]}")
print(f"P10 (first 3): {result['p10'][:3]}")
print(f"P90 (first 3): {result['p90'][:3]}")