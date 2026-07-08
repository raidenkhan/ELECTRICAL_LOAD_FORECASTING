"""Step 1: Generate D+1, D+7, D+30 predictions + actuals. Run with venv python."""
import sys, os, json, csv
sys.path.insert(0, '.')
os.environ['APP_SETTINGS'] = 'test'

import sqlite3
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from app.ml.dlinear_engine import DLinearEngine

# Load engine
engine = DLinearEngine()
print(f"Engine fitted: {engine.is_fitted}, models: {len(engine.models)}")

# Connect to DB
db_path = os.path.join(os.path.dirname(__file__), "..", "loadforecast.db")
db = sqlite3.connect(db_path)

# Get actuals from ecg_historical_demand (last 60 days available)
actuals = pd.read_sql("""
    SELECT date, hour, demand_mw, temperature_c
    FROM ecg_historical_demand
    WHERE date >= '2026-03-01'
    ORDER BY date, hour
""", db)
db.close()

actuals['datetime'] = pd.to_datetime(actuals['date'].astype(str)) + pd.to_timedelta(actuals['hour'] - 1, unit='h')
print(f"Actuals: {actuals['date'].min()} to {actuals['date'].max()}, {len(actuals)} rows")

# We'll generate predictions rolling day by day for a 30-day window
# For each test_date, predict D+1, D+7, D+30 and compare with actuals

test_dates = pd.date_range('2026-04-01', '2026-04-05', freq='D')
results = []

for test_date in test_dates:
    test_d = test_date.date()
    print(f"\n--- {test_d} ---")
    
    # Fetch history: 200 hours before test_date
    cutoff = test_date - timedelta(hours=200)
    hist = actuals[actuals['datetime'] < test_date].tail(200).copy()
    if len(hist) < 168:
        print(f"  Skip: only {len(hist)} history rows")
        continue
    
    hist = hist.rename(columns={'datetime': 'date'})
    
    # D+1: predict 24h starting at test_date midnight
    try:
        # Get future temps from actuals for same hours
        d1_end = test_date + timedelta(hours=24)
        d1_actuals = actuals[(actuals['datetime'] >= test_date) & (actuals['datetime'] < d1_end)]
        
        if len(d1_actuals) >= 20:
            future_temps = d1_actuals['temperature_c'].tolist()[:24]
            pred = engine.predict(hist, horizon_hours=24, future_temps_c=future_temps, use_tide=False)
            results.append({
                'test_date': str(test_d), 'horizon': 'D+1',
                'pred_mw': json.dumps(pred['forecast_mw']),
                'actual_mw': json.dumps(d1_actuals['demand_mw'].tolist()[:24]),
                'n_hours': min(len(d1_actuals), 24)
            })
            print(f"  D+1: pred={len(pred['forecast_mw'])}h, actuals={len(d1_actuals)}h")
    except Exception as e:
        print(f"  D+1 error: {e}")
    
    # D+7: predict 168h starting at test_date
    try:
        d7_end = test_date + timedelta(hours=168)
        d7_actuals = actuals[(actuals['datetime'] >= test_date) & (actuals['datetime'] < d7_end)]
        
        if len(d7_actuals) >= 120:
            future_temps = d7_actuals['temperature_c'].tolist()[:168]
            pred = engine.predict(hist, horizon_hours=168, future_temps_c=future_temps, use_tide=False)
            results.append({
                'test_date': str(test_d), 'horizon': 'D+7',
                'pred_mw': json.dumps(pred['forecast_mw']),
                'actual_mw': json.dumps(d7_actuals['demand_mw'].tolist()[:168]),
                'n_hours': min(len(d7_actuals), 168)
            })
            print(f"  D+7: pred={len(pred['forecast_mw'])}h, actuals={len(d7_actuals)}h")
    except Exception as e:
        print(f"  D+7 error: {e}")
    
    # D+30: predict 720h starting at test_date
    try:
        d30_end = test_date + timedelta(hours=720)
        d30_actuals = actuals[(actuals['datetime'] >= test_date) & (actuals['datetime'] < d30_end)]
        
        if len(d30_actuals) >= 480:
            future_temps = d30_actuals['temperature_c'].tolist()[:720]
            pred = engine.predict(hist, horizon_hours=720, future_temps_c=future_temps, use_tide=False)
            results.append({
                'test_date': str(test_d), 'horizon': 'D+30',
                'pred_mw': json.dumps(pred['forecast_mw']),
                'actual_mw': json.dumps(d30_actuals['demand_mw'].tolist()[:720]),
                'n_hours': min(len(d30_actuals), 720)
            })
            print(f"  D+30: pred={len(pred['forecast_mw'])}h, actuals={len(d30_actuals)}h")
    except Exception as e:
        print(f"  D+30 error: {e}")

# Save
out = pd.DataFrame(results)
out.to_csv('report_forecast_data.csv', index=False)
print(f"\nSaved {len(out)} rows to report_forecast_data.csv")
