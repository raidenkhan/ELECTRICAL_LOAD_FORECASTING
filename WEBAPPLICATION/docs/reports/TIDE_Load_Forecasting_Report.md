# TIDE Load Forecasting Report

**Date:** 2026-06-05

## Overview

The grid grew from 1,692 MW average demand in 2018 to 3,275 MW in 2026 — a **94% increase**. Most load forecasting research comes from Europe and the US, where grids grow at 1-2% per year. Our grid grows at 6-14% per year, which means a model trained last year is already outdated.

We built a day-ahead load forecasting system using DLinear (a lightweight neural network with ~40,000 parameters). The baseline model achieves **3.6% MAPE** — better than the operator's existing system — but we found a persistent problem: the model systematically **under-forecasts by 15-20 MW**, and this bias grows between retraining cycles.

We investigated why and tested 12 different approaches to fix it. Most failed. The one that worked was the simplest: track the recent forecast errors and subtract them from future predictions. We call this **TIDE** (Temporal Integration of Drift Errors).

## How TIDE Works (In Plain English)

1. Every day, the model makes a 24-hour forecast
2. We compare it to what actually happened and calculate the error
3. We keep a running average of recent errors (exponentially weighted — recent errors count more)
4. We subtract this average from tomorrow's forecast

That's it. Zero new parameters, zero extra training, about 20 lines of code.

The key insight: the forecast errors aren't random — they drift slowly over days and weeks. Think of the error as a low hum (predictable, slowly changing) mixed with static crackling (random noise). TIDE filters out the static and tracks the hum.

## Performance

| Metric | Value |
|--------|-------|
| Baseline MAE (average of 6 test years) | **93.9 MW** |
| + TIDE | **75.9 MW** |
| Improvement | **-19.2%** |
| Systematic bias before TIDE | ~18 MW |
| Systematic bias after TIDE | < 3 MW |

Every single test year improved — from 2021 to 2026-H1. The improvement was largest on the most recent year (2026, -20.9%), which is exactly where the model has drifted the most since its last retraining.

## How It Compares to Alternatives

| Method | MAE |
|--------|:---:|
| No correction (baseline) | 93.9 MW |
| Simple Moving Average (7-day) | 77.8 MW |
| Linear Trend extrapolation | 79.8 MW |
| Kalman Filter | 83.8 MW |
| **TIDE (our method)** | **75.9 MW** |

TIDE beats all three alternatives with a single setting (alpha = 0.3) that doesn't need tuning.

## Does It Work on Other Models?

Yes — we tested TIDE on 11 different model types (LSTM, Transformer, GRU, LightGBM, ARIMA, and more). Every single one improved by **23-26%**. The improvement is the same regardless of model complexity because TIDE fixes a problem in the data (the growth trend), not a problem specific to any one model.

## How Much Training Data Do You Actually Need?

The 6-fold cross-validation trained on up to 8 years of history. But we found that **2 years is almost as good as 8**:

| Training Window | Raw MAE | + TIDE |
|:---------------|:------:|:------:|
| 2 years (2024-2025) | 120.1 MW | 96.6 MW |
| 4 years (2022-2025) | 118.4 MW | 95.3 MW |
| 8 years (2018-2025) | 118.3 MW | 95.3 MW |

The 6-fold ensemble was a research tool for testing across different years. **For production, a single model trained on the 2-4 most recent years + TIDE is sufficient.** No need to train and maintain 6 separate models.

## Key Achievements

- **19.2% MAE reduction** with zero training, zero parameters, ~20 lines of code
- Works on **all 11 architectures tested** — LSTM, Transformer, LightGBM, ARIMA, etc.
- **Beats SMA, Kalman, and Linear Trend** correctors
- **95% confidence intervals** from 10,000 bootstrap samples confirm statistical significance (p < 0.001)
- **2-4 years of training data is enough** — more history doesn't help
- Full **6-fold cross-validation** across 8 years of growth (1,692 MW → 3,275 MW)

## Next Steps

- Run TIDE in production and monitor rolling MAE over 3-6 months
- Retrain baseline model annually (or when MAE degrades >10%)
- Deploy as a lightweight service — no GPU required, runs on CPU in < 1 second per day
- Write up the Sobolev trajectory loss results (Appendix B of the paper) as a separate technical note
- Prepare submission to *Applied Energy* or *Energy & AI*
