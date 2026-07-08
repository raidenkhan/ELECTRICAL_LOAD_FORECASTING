# DLinear + H10 Adaptive Level Correction

## Model Architecture

```
                    ┌──────────────────┐
                    │   Input x(t)     │
                    │  (168h window,   │
                    │   8 features)    │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐ ┌────▼────┐ ┌───────▼───────┐
       │   Moving    │ │ Seasonal│ │  Calendar     │
       │   Average   │ │  x - MA │ │  Features     │
       │  (kernel=25)│ │         │ │  (7 cols)     │
       └──────┬──────┘ └────┬────┘ └───────┬───────┘
              │             │              │
              ▼             ▼              ▼
       ┌──────────────┐ ┌────────┐ ┌──────────────┐
       │ Trend Linear │ │Seasonal│ │ Calendar     │
       │ 168 ➔ 24     │ │168 ➔ 24│ │ Linear       │
       └──────┬───────┘ └───┬────┘ │ 1176 ➔ 24    │
              │             │      └──────┬───────┘
              └─────────────┴─────────────┘
                              │
                     ┌────────▼────────┐
                     │   Ensemble      │
                     │   (6 models)    │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │   H10 Bias      │
                     │   Correction    │
                     │  (EMA, α=0.3)   │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │   Output y(t)   │
                     │   (24h forecast) │
                     └─────────────────┘
```

## DLinear Component

The DLinear model decomposes the input time series into trend and seasonal components:

1. **Moving Average** (kernel=25): extracts the slow-moving trend
2. **Seasonal** = input - trend: captures periodic patterns
3. **Calendar features** (hour_sin/cos, dow_sin/cos, month_sin/cos, temperature_c): external predictors

Each component passes through a linear layer mapping 168h ➔ 24h, then summed.

## 6-Fold Ensemble

Six models trained on expanding windows (all data 2018-2025):

| Fold | Train | Test | Test MAE |
|------|-------|------|----------|
| Fold_1 | 2018-2020 | 2021 | 77.3 MW |
| Fold_2 | 2018-2021 | 2022 | 81.6 MW |
| Fold_3 | 2018-2022 | 2023 | 86.9 MW |
| Fold_4 | 2018-2023 | 2024 | 93.6 MW |
| Fold_5 | 2018-2024 | 2025 | 103.0 MW |
| Fold_6 | 2018-2025 | 2026-H1 | 120.7 MW |

Ensemble prediction = mean of 6 individual forecasts. Reduces variance and improves robustness.

## H10 Adaptive Level Correction

Online bias corrector that learns from recent errors:

1. Stores last N (prediction, actual) pairs in a buffer
2. Computes EMA-smoothed bias: `bias(t) = α × mean(error) + (1-α) × bias(t-1)`
3. Applies correction: `y_corrected = y_raw + bias`

- α = 0.3 (smoothing factor)
- Window = 48 hours (2 days of errors)
- Persisted to SQLite between restarts

## Normalization

All 8 features are z-score normalized per fold:

```
x_norm = (x - μ_fold) / σ_fold
```

The engine auto-selects the **last fold** (Fold_6: μ=2199, σ=443) for normalization — this best matches current (2026) conditions.

## Training

- Optimizer: Adam (lr=0.001)
- Loss: L1 (MAE)
- Batch size: 4096
- Early stopping: patience=15 epochs
- Max epochs: 200
- Training time: ~18 minutes (CPU, 6 folds)
