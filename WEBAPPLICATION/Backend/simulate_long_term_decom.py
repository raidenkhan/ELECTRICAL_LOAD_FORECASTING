import os
import pandas as pd
import numpy as np
import joblib
import sys

# Add backend to path
sys.path.append(os.getcwd())

from app.ml.decom_engine import DecomEngine
from app.core.config import settings

def simulate_3_months():
    model_path = "models/decomp_engine.joblib"
    if not os.path.exists(model_path):
        print("Model not found. Run init_decom_model.py first.")
        return

    engine = DecomEngine()
    engine.load(model_path)
    
    # Verify nudge (Manual simulation of SCADA reality)
    engine.trend.nudge_trend(157.0, pd.Timestamp.now().date())
    
    # Simulate 90 days of timestamps
    start_ts = pd.Timestamp.now().floor('15min')
    end_ts = start_ts + pd.Timedelta(days=90)
    future_ts = pd.date_range(start=start_ts, end=end_ts, freq='15min')
    
    df_future = pd.DataFrame(index=future_ts)
    df_future['DATETIME'] = df_future.index
    df_future['Date'] = df_future.index.date
    df_future['TimeSlot'] = df_future.index.hour * 4 + df_future.index.minute // 15
    df_future['DOW'] = df_future.index.dayofweek
    df_future['Is_Holiday'] = 0
    
    # Simulate realistic temperature variations (24C to 38C)
    # Peak at 14:00 daily
    hour = future_ts.hour + future_ts.minute / 60.0
    df_future['Temp'] = 28.0 + 6.0 * np.sin(2 * np.pi * (hour - 8) / 24)
    
    # Precip (mostly dry for baseline check)
    df_future['precip_mm'] = 0.0

    print(f"Running simulation for 90 days ({len(future_ts)} steps)...")
    prediction = engine.predict(df_future)
    forecast = np.array(prediction["forecast_mw"])
    
    print("\n--- Simulation Results ---")
    print(f"Minimum Predicted: {forecast.min():.2f} MW")
    print(f"Maximum Predicted: {forecast.max():.2f} MW")
    print(f"Mean Predicted:    {forecast.mean():.2f} MW")
    
    # Check for specific thresholds
    over_120 = np.sum(forecast >= 120.0)
    print(f"Steps >= 120 MW:   {over_120} ({over_120/len(forecast)*100:.2f}%)")
    
    if over_120 > 0:
        peak_idx = np.argmax(forecast)
        print(f"Peak predicted at: {future_ts[peak_idx]} ({forecast[peak_idx]:.2f} MW)")
    else:
        print("Model NEVER reaches 120 MW in this 3-month simulation.")

if __name__ == "__main__":
    simulate_3_months()
