"""Evaluate WT+DOW D+1 MAPE per calendar month (Jan, Feb, Mar 2026)."""

import pandas as pd
import numpy as np
import sys
from datetime import date, timedelta
from app.ml.weighted_trend_engine import WeightedTrendEngine

DATA = r'C:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\data\ecg_actual_demand_clean_with_temp.csv'
MODEL = r'C:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\WEBAPPLICATION\Backend\models\weighted_trend_engine.joblib'

df = pd.read_csv(DATA)
df['Date'] = pd.to_datetime(df['date'])
df['Hour'] = df['hour'].astype(int)
df['demand_mw'] = df['demand_mw'].astype(float)

engine = WeightedTrendEngine()
engine.load(MODEL)

months = [1, 2, 3]  # Jan, Feb, Mar 2026
year = 2026

for month in months:
    # Seed history with all data BEFORE this month
    cutoff = date(year, month, 1)
    hist = df[df['Date'] < pd.Timestamp(cutoff)]
    engine.load_history(hist)
    # _last_daily_means is sorted by date index

    actual_month = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month)].copy()
    days = sorted(actual_month['Date'].dt.date.unique())
    daily_mapes = []

    for target_date in days:
        result = engine.predict_for_date(target_date)
        forecast = np.array(result['forecast_mw'])

        actual = actual_month[actual_month['Date'].dt.date == target_date]['demand_mw'].values
        if len(actual) != 24:
            continue

        ape = np.abs((actual - forecast) / actual) * 100
        daily_mapes.append(np.mean(ape))

        # Update _last_daily_means so tomorrow's forecast sees today's actual
        engine._last_daily_means[target_date] = np.mean(actual)
        engine._last_daily_means = engine._last_daily_means.sort_index()

    mean_mape = np.mean(daily_mapes) if daily_mapes else 0
    print(f"2026-{month:02d}: {mean_mape:.2f}% MAPE ({len(daily_mapes)} days, sd={np.std(daily_mapes):.2f})")

print("Done.")
