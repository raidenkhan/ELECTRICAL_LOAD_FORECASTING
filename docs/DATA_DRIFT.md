# Data Drift Analysis

## Year-over-Year Demand Growth

ECG (Ghana) demand has grown ~94% from 2018 to 2026:

![Demand Growth](../results/7day_forecast.png)

| Year | Mean (MW) | Std (MW) | Min (MW) | Max (MW) | YoY Growth |
|------|-----------|----------|----------|----------|------------|
| 2018 | 1,692 | 195 | 1,224 | 2,611 | — |
| 2019 | 1,803 | 222 | 1,199 | 3,147 | +6.6% |
| 2020 | 1,989 | 232 | 1,064 | 3,216 | +10.3% |
| 2021 | 2,203 | 246 | 1,361 | 3,300 | +10.8% |
| 2022 | 2,258 | 282 | 1,281 | 3,511 | +2.5% |
| 2023 | 2,379 | 275 | 1,527 | 3,548 | +5.4% |
| 2024 | 2,599 | 297 | 1,550 | 3,537 | +9.2% |
| 2025 | 2,879 | 352 | 1,789 | 3,943 | +10.8% |
| 2026* | 3,275 | 346 | 1,905 | 4,117 | +13.8% |

*\*Partial year (Jan-May)*

## Impact on Model

### Pre-Retrain (old Fold_5 stats: μ=2,054, σ=347)
- 2026 demand z-score: (3,275 - 2,054) / 347 = **3.51σ** above training mean
- Model operates at the extreme edge of its training distribution
- Normalization mismatch leads to degraded accuracy

### Post-Retrain (new Fold_6 stats: μ=2,199, σ=443)
- 2026 demand z-score: (3,275 - 2,199) / 443 = **2.43σ** above training mean
- Still elevated but within the model's learned range
- Retrain on full 2018-2026 dataset ensures the model has seen recent patterns

## Retrain Strategy

### Schedule
- **Every 6 months** (or when rolling MAE degrades >10%)
- Current baseline: **67 MW** D+1 MAE (from stress tests)
- Trigger: rolling 30-day MAE > 73.7 MW

### Detection
The `MetricsService.check_drift()` method compares the rolling MAE against the baseline:

```
degradation = (current_mae - baseline_mae) / baseline_mae
drifted = degradation > 0.10 (10% threshold)
```

### Retrain Process
```
tools/retrain_dlinear.py
  1. Load data from CSV (2018-2026)
  2. Train 6 folds (expanding windows)
  3. Save checkpoints to models/dlinear/
  4. Save normalization_stats.json
  5. Restart server to pick up new models
```

Total time: ~18 minutes on CPU (all 6 folds).

## H10 Online Adaptation

Between retrains, the H10 corrector provides online adaptation:

- Stores the last 48 hours of (predicted, actual) pairs
- Computes EMA-smoothed bias
- Persisted to SQLite across restarts

This catches short-term shifts (weather, holidays, weekdays) that the DLinear ensemble misses.
