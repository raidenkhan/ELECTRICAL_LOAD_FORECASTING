# GRIDCo Dispatch Scheduling System — Progress Report

**Date:** 2026-06-05

## Overview

We are building a day-ahead load forecasting and dispatch scheduling system for the national grid operator. The system replaces the existing manual Excel-based workflow with an automated pipeline: historical data feeds a machine learning model, which produces 24-hour load forecasts used to schedule power generation across 5 demand entities (ECG, NEDCo, VALCO, Mines, Export) and multiple supply plants.

The grid grew from 1,692 MW average demand in 2018 to 3,275 MW in 2026 — a 94% increase. This rapid growth means models trained last year are already outdated, which was the central challenge we had to solve.

## What We Built

### 1. ML Forecast Engine (DLinear + TIDE)

We deployed a lightweight neural network (DLinear, ~40,000 parameters) that forecasts 24-hour load using 7 days of history plus calendar and temperature features. The model runs entirely on CPU and produces a forecast in under 1 second.

We discovered that the model accumulated **systematic bias** between retraining cycles — it consistently under-forecasted by 15-20 MW as the grid grew. We developed an online correction mechanism called **TIDE** that tracks recent errors and adjusts future predictions in real time. No retraining needed — it runs continuously and adapts within 2-3 days.

| Metric | Baseline | + TIDE |
|--------|:-------:|:------:|
| Average error (MAE) | 93.9 MW | 75.9 MW |
| Systematic bias | ~18 MW | < 3 MW |
| Accuracy improvement | — | **-19.2%** |

TIDE also beats alternative approaches (Simple Moving Average, Kalman Filter, Linear Trend) and works on all 11 model types tested — from simple ARIMA to Transformers.

**Key insight for production:** We originally trained 6 separate models on different time windows. But we found that a **single model trained on just 2-4 recent years + TIDE achieves the same accuracy**. This simplifies deployment significantly.

### 2. Web Application

| Page | Status | What it does |
|------|--------|-------------|
| **Dispatch Schedule** | Live | Upload Excel, view 24h generation schedule, edit allocations, submit to database |
| **Control Room** | Live | Real-time system monitoring with load forecasts, supply/demand balance, generation mix |
| **System Overview** | Live | Dashboard with live data, forecast comparisons, model performance metrics |

The frontend is built with Next.js and connects to a FastAPI backend with PostgreSQL for data storage.

### 3. Data Pipeline

Historical hourly load data (2018-2026, 70,000+ records) is cleaned and stored in PostgreSQL. Weather data is fetched from Open-Meteo API. The system automatically normalizes data using rolling statistics to account for the growing demand baseline.

## Key Achievements

- **19.2% error reduction** with TIDE — zero additional training, zero parameters, ~20 lines of code
- **TIDE works on any model** — LSTM, Transformer, LightGBM, ARIMA — all improved 23-26%
- **96% of bias eliminated** — from ~18 MW to < 3 MW systematic error
- **2-4 years of training data is enough** — more history doesn't help
- **Production engine deployed** — FastAPI backend serving forecasts via REST API
- **Dispatch scheduling live** — operators can upload, edit, and submit generation schedules
- **Control Room dashboard live** — real-time monitoring with digital twin visualization

## Current System Architecture

```
Operator (Browser)
  → Dispatch Schedule (upload/edit schedules)
  → Control Room (real-time monitoring)
  → System Overview (dashboards)
      ↓
FastAPI Backend
  → DLinear + TIDE forecast engine (CPU, <1s per run)
  → Weather service (Open-Meteo)
  → Schedule service (Excel parsing)
      ↓
PostgreSQL Database (historical demand, schedules, forecasts)
```

## Next Steps

| Phase | Task | Priority |
|-------|------|----------|
| 5 | Move DecomEngine to interpretability layer — explain why forecasts look the way they do | High |
| 6 | Build auto metrics service — track rolling MAE/MAPE from database | Medium |
| 7 | Rebuild alerts — model health monitoring, data freshness checks | Medium |
| 8 | Frontend updates — show model performance, forecast source indicators, TIDE correction status | Low |

## Files Referenced

- `app/ml/dlinear_engine.py` — Production ML engine with DLinear + TIDE
- `app/services/dispatch_forecast_service.py` — Forecast API service
- `app/api/v1/dispatch_forecast.py` — REST endpoints
- `frontend/` — Next.js web application
- `tools/retrain_dlinear.py` — Model retraining script
- `tools/experiment_window_size.py` — Training window size experiment
