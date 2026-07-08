# Phase 2 — Synthetic ECG Demand Data

## Overview

Phase 2 uses **synthetic ECG hourly demand data** for the DecomEngine forecasting model (24-step, weather-aware, DB-driven growth). The synthetic data was generated from a **May 22 profile** with day-of-week, seasonal, and temperature variation.

## Data Source

- **Generator**: `Backend/tools/seed_ecg_data.py`
- **Duration**: 387 days (9,288 hourly records)
- **Base profile**: Derived from the May 22 ECG demand pattern
- **Synthetic parameters**:
  - `TEMP_SENSITIVITY` = 1.5% / °C
  - `SEASONAL_FACTOR` range: 0.92–1.05
  - `DOW_FACTOR` range: 0.88–1.02

## Model

- **Engine**: `Backend/app/ml/decom_engine_hourly.py`
- **Saved model**: `Backend/models/decomp_engine_hourly.joblib`
- **Training script**: `Backend/tools/train_ecg_engine.py`
- **Gate test**: `Backend/tools/test_ecg_engine.py` — Phase 2 gate test PASSED (holdout 7-day MAE = 63.0 MW, threshold < 150 MW)

## Replacement Instructions

When 12+ months of real ECG hourly demand + temperature data becomes available:

1. Prepare the real data as a CSV with columns: `timestamp`, `demand_mw`, `temperature_c`
2. Run `python tools/train_ecg_engine.py` with the real file
3. Verify gate test passes with the new model
4. Update `dispatch_forecast_service.py` if column names differ
5. Delete this file once verified
