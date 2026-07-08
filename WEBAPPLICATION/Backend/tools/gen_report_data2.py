"""Generate D+1, D+7, D+30 predictions + actuals. Run with venv python."""
import sys, os, json
sys.path.insert(0, '.')
os.environ['APP_SETTINGS'] = 'test'

import sqlite3
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from app.ml.dlinear_engine import DLinearEngine

engine = DLinearEngine()
print(f"Engine fitted: {engine.is_fitted}, models: {len(engine.models)}")

# Connect to DB
db_path = os.path.join(os.path.dirname(__file__), "..", "loadforecast.db")
con = sqlite3.connect(db_path)
df_raw = pd.read_sql("""
    SELECT date, hour, demand_mw, temperature_c
    FROM ecg_historical_demand
    WHERE date >= '2025-01-01'
    ORDER BY date, hour
""", con)
con.close()

# Build proper datetime index
df_raw['ts'] = pd.to_datetime(df_raw['date'].astype(str)) + pd.to_timedelta(df_raw['hour'] - 1, unit='h')
df_raw = df_raw.sort_values('ts').reset_index(drop=True)

print(f"Data: {df_raw['date'].min()} to {df_raw['date'].max()}, {len(df_raw)} rows")

# Generate forecasts for a few days
test_dates = pd.date_range('2026-04-01', '2026-04-03', freq='D')
results = []

for test_date in test_dates:
    test_d = test_date.date()
    print(f"\n--- {test_d} ---")
    
    # History: 200 hours before test_date midnight
    cutoff_ts = pd.Timestamp(test_date)
    hist = df_raw[df_raw['ts'] < cutoff_ts].tail(200).copy()
    hist_df = hist[['ts', 'demand_mw', 'temperature_c']].rename(columns={'ts': 'date'})
    
    if len(hist_df) < 168:
        print(f"  Skip: only {len(hist_df)} history rows")
        continue
    
    # Helper: get future slice
    def get_future_slice(hours):
        end = cutoff_ts + timedelta(hours=hours)
        fut = df_raw[(df_raw['ts'] >= cutoff_ts) & (df_raw['ts'] < end)]
        return fut['demand_mw'].tolist()[:hours], fut['temperature_c'].tolist()[:hours]
    
    # D+1
    actuals_24, temps_24 = get_future_slice(24)
    if len(actuals_24) >= 20:
        try:
            pred = engine.predict(hist_df, horizon_hours=24, future_temps_c=temps_24, use_tide=False)
            results.append({'test_date': str(test_d), 'horizon': 'D+1',
                'pred_mw': json.dumps(pred['forecast_mw']),
                'actual_mw': json.dumps(actuals_24),
                'n_hours': len(actuals_24)})
            print(f"  D+1: MAE={sum(abs(p-a) for p,a in zip(pred['forecast_mw'],actuals_24))/len(actuals_24):.0f} MW")
        except Exception as e:
            print(f"  D+1 err: {e}")
    
    # D+7
    actuals_168, temps_168 = get_future_slice(168)
    if len(actuals_168) >= 120:
        try:
            pred = engine.predict(hist_df, horizon_hours=168, future_temps_c=temps_168, use_tide=False)
            results.append({'test_date': str(test_d), 'horizon': 'D+7',
                'pred_mw': json.dumps(pred['forecast_mw']),
                'actual_mw': json.dumps(actuals_168),
                'n_hours': len(actuals_168)})
            print(f"  D+7: MAE={sum(abs(p-a) for p,a in zip(pred['forecast_mw'],actuals_168))/len(actuals_168):.0f} MW")
        except Exception as e:
            print(f"  D+7 err: {e}")
    
    # D+30
    actuals_720, temps_720 = get_future_slice(720)
    if len(actuals_720) >= 480:
        try:
            pred = engine.predict(hist_df, horizon_hours=720, future_temps_c=temps_720, use_tide=False)
            results.append({'test_date': str(test_d), 'horizon': 'D+30',
                'pred_mw': json.dumps(pred['forecast_mw']),
                'actual_mw': json.dumps(actuals_720),
                'n_hours': len(actuals_720)})
            print(f"  D+30: MAE={sum(abs(p-a) for p,a in zip(pred['forecast_mw'],actuals_720))/len(actuals_720):.0f} MW")
        except Exception as e:
            print(f"  D+30 err: {e}")

out = pd.DataFrame(results)
out.to_csv(os.path.join(os.path.dirname(__file__), "..", "report_forecast_data.csv"), index=False)
print(f"\nSaved {len(out)} rows")
