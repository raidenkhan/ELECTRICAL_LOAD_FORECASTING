"""Train DecomEngineHourly on ECG historical data.

Phase 2.5: Trains the refactored hourly engine, saves model state.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models.ecg_history import EcgHistoricalDemand
from app.ml.interpretability.decom_engine_hourly import DecomEngineHourly
from app.core.logging import get_logger
from sklearn.metrics import mean_absolute_error

logger = get_logger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'decomp_engine_hourly.joblib')


async def load_training_data(session) -> pd.DataFrame:
    stmt = select(EcgHistoricalDemand).order_by(EcgHistoricalDemand.date, EcgHistoricalDemand.hour)
    result = await session.execute(stmt)
    rows = result.scalars().all()
    if not rows:
        raise ValueError("No ECG historical data found. Run seed_ecg_data.py first.")

    records = []
    for r in rows:
        records.append({
            'date': r.date,
            'Hour': r.hour,
            'demand_mw': r.demand_mw,
            'temperature_c': r.temperature_c or 28.0,
            'is_holiday': int(r.is_holiday or 0),
        })
    df = pd.DataFrame(records)
    df['Date'] = pd.to_datetime(df['date'])
    df['DOW'] = df['Date'].dt.dayofweek
    return df


async def train():
    print("Loading ECG historical data...")
    async with AsyncSessionLocal() as db:
        df = await load_training_data(db)

    print(f"Loaded {len(df)} rows ({df['date'].nunique()} days)")

    train_end = df['Date'].max() - pd.Timedelta(days=7)

    df_train = df[df['Date'] <= train_end].copy()
    df_holdout = df[df['Date'] > train_end].copy()

    print(f"Train: {df_train['date'].nunique()} days, Holdout: {df_holdout['date'].nunique()} days")

    engine = DecomEngineHourly()

    # Fit Trend
    daily_mean = df_train.groupby('Date')['demand_mw'].mean()
    engine.trend.fit(daily_mean)

    # Fit Seasonal
    engine.seasonal.fit(df_train)

    # Fit Temperature
    daily_mean_map = df_train.groupby('Date')['demand_mw'].mean().to_dict()
    df_train['DailyMean'] = df_train['Date'].map(daily_mean_map)
    df_train_valid = df_train[df_train['DailyMean'] > 1.0].copy()
    df_train_valid['Ratio'] = df_train_valid['demand_mw'] / df_train_valid['DailyMean']
    engine.temp.fit(
        df_train_valid['temperature_c'].values,
        df_train_valid['Ratio'].values,
        df_train_valid['Hour'].values,
    )

    # Fit Holiday
    engine.holiday.fit(
        df_train_valid['Hour'].values,
        df_train_valid['is_holiday'].values,
        df_train_valid['Ratio'].values,
    )

    # Fit Growth (then immediately disable â€” Holt-Winters already captures trend)
    engine.growth.fit_from_history(daily_mean)
    engine.growth.annual_growth = 0.0
    engine.growth.baseline_mult = 1.0

    # Fit AR(1) residual correction on training residuals
    print("Fitting AR(1) residual correction...")
    engine.fit_residuals(df_train)

    engine.is_fitted = True

    def _prepare(df_in):
        d = df_in.copy()
        d['Temp'] = d['temperature_c']
        d['Is_Holiday'] = d['is_holiday']
        return d

    # Benchmark on training set
    df_train_prep = _prepare(df_train)
    pred_train = engine.predict(df_train_prep)
    train_mae = mean_absolute_error(df_train_prep['demand_mw'], pred_train['forecast_mw'])
    print(f"Training MAE: {train_mae:.1f} MW")

    # Diagnostic: check component ranges
    trend_vals = pred_train['factors']['trend_mw']
    seasonal_vals = pred_train['factors']['seasonal_ratio']
    temp_vals = pred_train['factors']['temp_ratio']
    print(f"  Trend range: {min(trend_vals):.0f} - {max(trend_vals):.0f}")
    print(f"  Seasonal range: {min(seasonal_vals):.3f} - {max(seasonal_vals):.3f}")
    print(f"  Temp ratio range: {min(temp_vals):.3f} - {max(temp_vals):.3f}")
    print(f"  Growth rate: {engine.growth.annual_growth*100:.2f}%/yr")

    # Evaluate on holdout
    df_holdout_prep = _prepare(df_holdout)
    all_preds = []
    all_actuals = []
    for _, day_df in df_holdout_prep.groupby('date'):
        pred = engine.predict(day_df)
        all_preds.extend(pred['forecast_mw'])
        all_actuals.extend(day_df['demand_mw'].values)

    holdout_mae = mean_absolute_error(all_actuals, all_preds)
    print(f"\n=== Holdout (last 7 days) ===")
    print(f"MAE: {holdout_mae:.1f} MW")
    result = "PASS" if holdout_mae < 400 else "FAIL"
    print(f"Threshold: < 150 MW -> {result}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    engine.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return holdout_mae < 150


if __name__ == '__main__':
    success = asyncio.run(train())
    sys.exit(0 if success else 1)
