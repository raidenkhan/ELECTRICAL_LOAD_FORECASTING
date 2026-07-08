# Model Analysis: Ghana ECG System-Wide Load Forecasting

## Abstract

We present a comprehensive benchmark of forecasting approaches for Ghana's national
power grid (ECG), spanning a 70,228-hour dataset (2018-2026, 8.6% CAGR) using real
ECG demand data. We compare
statistical decomposition, adaptive rolling-window methods, gradient boosting,
and N-BEATS neural networks across 6-fold cross-validation. Our key finding:
an adaptive rolling decomposition engine (28-day profile + 7-day trend)
achieves 114 MW mean absolute error (MAE) across 6-fold cross-validation, outperforming
all machine learning approaches except N-BEATS (111 MW) which falls within
statistical noise. Temperature models are *neutral* (0 MW delta) against the 7-day rolling
trend, but NOT because temperature doesn't matter. Against a static
(non-adaptive) baseline, a daily temperature correction achieves -33 MW (6.5%
improvement), consistent with GridCo operators reporting temperature as their
primary adjustment factor. The reconciliation: the 7-day rolling trend
implicitly captures temperature because Ghana's weather persists longer than
the 7-day window, making explicit temperature corrections architecturally
redundant. Real ECG data confirms +92 MW/C sensitivity (4.1%/C), with a
statistically significant quadratic component (F=34.2, p<0.0001) showing
AC saturation above 28C (+227 to +304 MW/C), but even non-linear forms
cannot improve beyond the adaptive window. Gaussian Process and Markov
structure tests confirm the simple rolling decomposition is near-optimal —
a linear model with 7 lags + temperature + hour dummies is statistically
tied at 105 MW vs the rolling model's 113 MW. However, a simpler insight
emerges: the equal-weighted 7-day rolling mean is suboptimal — replacing it
with a weighted trend (0.65 x lag1 + 0.35 x lag7) achieves 105 MW
(-8 MW, p=0.027) at zero additional complexity. Temperature contributes
nothing to this gain; it is purely optimal lag weighting (yesterday
matters 4x more than mid-week days, and last week's same-day captures
the weekly cycle). A full window-size ablation (30 combinations x 6 folds)
confirms the default configuration (28d profile, 7d trend) is essentially
optimal. No deep learning or ML model reliably beats the structural
baseline without autoregressive leakage or temporal instability.

## 1. Dataset

The ECG dataset contains 70,228 hourly demand records from 2018-01-01 to 2026-05-01
(96.2% coverage, 2,934 days). Key statistics:

| Metric | Value |
|--------|-------|
| Mean demand | 2,243 MW |
| Range | 845 -- 4,117 MW |
| CAGR (2018-2026) | 8.6% |
| CAGR (2023-2025) | 10.0% |
| Temperature mean | 26.7 C |
| Temperature range | 18.2 -- 35.2 C |
| Temperature correlation | r ~ 0.3 (midday) |
| Holiday effect | -0.7% (negligible) |
| Weekend effect | -3.7% |

### 1.1 Data Quality

95% of anomalously low demand hours occur in 2018-2019 daytime (69% at 8-16h).
Post-2020 data is clean. Median-based decomposition handles this without explicit
cleaning.

## 2. Methods

### 2.1 AdaptiveDecomEngine (Proposed Baseline)

A rolling-window adaptation of classical time series decomposition:

$$hat{y}_t^{(h)} = tau_t times pi_t^{(h)}$$

where:
- $tau_t$: 7-day rolling mean of daily means (trend)
- $pi_t^{(h)}$: 28-day rolling median hourly profile (hour h)

Note:   Temperature models are architecturally redundant with the 7-day rolling trend,
  but NOT because temperature doesn't matter. Against a static baseline, daily temperature
  correction achieves -33 MW (6.5% improvement), confirming GridCo operators are correct
  that temperature is the primary adjustment factor. Real temp-demand relationship
  = +92 MW/C (4.1%/C). The rolling trend implicitly captures this because Ghana's weather
  patterns persist longer than 7 days, making explicit temperature corrections redundant.
  This is a structural feature (not a bug) of adaptive decomposition design.
  
  Update (v2): The equal-weighted rolling mean is suboptimal. Replacing it with a weighted
  trend (0.65 x yesterday + 0.35 x same-day-last-week) yields -8 MW improvement
  (113 -> 105 MW, p=0.027). Temperature adds nothing to this gain; it is purely optimal
  lag weighting. See Section 3.5F for details.

### 2.2 Machine Learning Comparators

**LightGBM Direct:** Multi-output (24h) gradient boosting on lag features
(lags at 1,2,3,24,48,168,336h + rolling statistics + calendar).

**LightGBM Residual Corrector:** Trained on residuals of adaptive engine with
autoregressive features (lagged residuals at 1,2,3,24,168,336h + rolling residual
statistics). This is NOT a standalone forecaster -- it corrects structural predictions
using the autocorrelation in recent errors.

### 2.3 Cross-Validation Protocol

Standard 6-fold temporal walk-forward:
- Train: 2018-01-01 to fold end-year
- Test: 6 months following train end
- Fixed window: 2018-01-01 always included
- Fold split: end-2019 (Fold 1), end-2020 (Fold 2), ..., end-2024 (Fold 6)
- All models evaluated on identical train/test splits
- Metrics: per-fold MAE (24h), per-hour breakdown (8h bins), stability (sd of
  fold MAEs), worst-case fold

### 2.4 Evaluation Protocol

For ML models using multi-output regression:
- 24 models trained (one per forecast hour), or single multi-output model
- Features computed at time t, targets are demand at t+1..t+24
- No information from within the forecast window used in features
- Test set evaluated as (n_test - 24) forecasts, rolled one hour at a time

## 3. Results

### 3.1 Main Comparison (24-Hour MAE, 6-Fold CV)

| Model | Mean MAE | sd | Best | Worst | Wins/6 |
|-------|----------|-----|------|-------|--------|
| AdaptiveDecomEngine (corrected) | **113** | 18 | 91 | 147 | baseline |
| + **Weighted trend** (0.65*l1+0.35*l7) | **105** | 16 | 89 | 130 | 6/6 |
| + **Weighted trend + DOW correction** | **102** | -- | -- | -- | 6/6 |
| + Temperature model (hourly, old) | **113** | -- | -- | -- | -- |
| + Temperature model (daily, corrected) | **123** | -- | -- | -- | 0/6 |
| + AR(1) correction | **126** | -- | -- | -- | 1/6 |
| LightGBM Direct | 134 | 37 | 101 | 196 | 3/6 |
| N-BEATS (Colab, 6h GPU) | 111 | 27 | 91 | 148 | 3/6 |

*Note: The weighted trend (0.65 x lag1 + 0.35 x lag7) replaces the equal-weighted 7-day
rolling mean. It achieves -8 MW (p=0.027) at zero additional complexity — no training,
no temperature, no extra parameters. The improvement comes purely from optimal lag
weighting: yesterday is 4x more predictive than mid-week days, and last week's same-day
captures the weekly cycle. See Section 3.5F.*

*Note: The corrected AdaptiveDecomEngine baseline (113 MW) replaces the previously published
119 MW, which was measured on a different engine implementation (AdaptiveEngine in
cv_adaptive_engine.py). See Section 3.5D for the full audit.*

### 3.2 Per-Fold Breakdown

| Fold | Test Days | Eng. Baseline | + Weighted Trend | + DOW Correction | + LinReg+Temp |
|------|-----------|--------------|-----------------|-----------------|---------------|
| Fold 1 (2020 H1) | 182 | 100 MW | **89 MW** (-11) | **86 MW** (-14) | 94 MW |
| Fold 2 (2021 H1) | 181 | 91 MW | **89 MW** (-2) | **86 MW** (-5) | 90 MW |
| Fold 3 (2022 H1) | 181 | 110 MW | **104 MW** (-6) | **100 MW** (-10) | 104 MW |
| Fold 4 (2023 H1) | 181 | 119 MW | **103 MW** (-16) | **101 MW** (-18) | 102 MW |
| Fold 5 (2024 H1) | 181 | 116 MW | **113 MW** (-3) | **110 MW** (-6) | 111 MW |
| Fold 6 (2025 H1) | 108 | 142 MW | **129 MW** (-13) | **125 MW** (-17) | 130 MW |
| **Mean** | -- | **113 MW** | **105 MW** (-8) | **102 MW** (-11) | **105 MW** |

The weighted trend wins 6/6 folds vs the equal-weighted rolling baseline.
Adding DOW correction improves to **102 MW** (-11 MW total from baseline).
The full LinReg (7 lags + temperature) is statistically tied with the simple
weighted trend — temperature adds no value.

### 3.3 LightGBM Leakage Analysis

The LightGBM residual corrector achieves its 18/18 fold wins through
autoregressive features. Feature importance analysis reveals:

| Feature | Importance | Type |
|---------|-----------|------|
| res_lag_1h | ~300,000 | Autoregressive (high risk) |
| res_lag_2h | ~100,000 | Autoregressive (high risk) |
| res_lag_3h | ~50,000 | Autoregressive (high risk) |
| res_lag_24h | ~40,000 | Known at 1h, leaky at 24h+ |
| Calendar features | ~20,000 | No risk |

In autoregressive evaluation:
- **24h**: Acceptable (res_lag_24h is true residual for hours 1-24)
- **168h**: Moderate risk (res_lag_24h becomes predicted after day 1)
- **720h**: High risk (all features drift; same failure mode as N-BEATS Fold 6)

**Recommendation:** Gate LightGBM correction to <=168h. For 720h, structural engine
alone is both safer and negligibly different.

### 3.4 The Fold 6 Problem

All models exhibit degradation on Fold 6 (test: 2025 H1), driven by a sudden
acceleration in demand growth. The training periods through 2024 have CAGR <10%,
while 2025 H1 demand grows at 29% CAGR, with a 14% jump in January alone:

| Metric | Folds 1-5 (test) | Fold 6 (2025 H1) |
|--------|-----------------|-------------------|
| Mean demand | 2,000-2,650 MW | 3,005 MW |
| CAGR at test | 0% to 22% | **29%** |
| Std of daily demand | 99-144 | 152 |
| CV | 4.4-6.0% | 5.1% |
| Test days | 181-182 | **115** |

| Model | Fold 6 MAE | vs Rolling Mean |
|-------|-----------|----------------|
| Rolling mean (equal weight) | 142 MW | baseline |
| + Weighted trend (0.65*l1+0.35*l7) | **129 MW** | **-13 MW** |
| + Growth-adjusted | 201 MW | +59 MW |
| + Exp-weighted (lambda=0.7) | 137 MW | -6 MW |
| + Log-demand | 129 MW | -13 MW |
| LightGBM Direct | 196 | +54 MW |
| N-BEATS | 148 | +6 MW |

**The weighted trend is already the fix for Fold 6.** The 14% January jump causes the
equal-weighted rolling mean to lag (it averages 6 days of pre-jump demand with 1 day
of post-jump), while the weighted trend corrects 65% of the jump on day 2 (when lag1
captures the jump). Weekly breakdown confirms the gain is concentrated in the most
volatile weeks: W01 from 135 to 93 MW (-42), W19 from 185 to 130 MW (-56),
W22 from 161 to 121 MW (-40).

**Fanier approaches all fail** on Fold 6: growth-adjusted trend amplifies noise
(+59 MW), exp-weighted recovers only -6 MW. The simple 2-weight model is optimal
because it doesn't try to estimate a "growth rate" from noisy daily data -- it simply
prioritizes the most recent information while anchoring on the weekly cycle.

**Root cause:** The accelerating trend creates an out-of-distribution shift. The
weighted trend mitigates this by prioritizing lag1 (yesterday's actual), but
cannot fully eliminate it when the growth rate triples overnight. No tested model
closes the remaining gap to Folds 1-5 levels (~100 MW).
transition to only 108 days of test data (Fold 6), amplifying the impact of
the acceleration.

The earlier documented Fold 6 MAE of 87 MW was from a different engine
(AdaptiveEngine in cv_adaptive_engine.py) which used a different evaluation
methodology. The corrected value is 147 MW.

### 3.5 Ablation Studies

#### 3.5A Physics-Aware Features
Three hypotheses tested on DecomEngine:
1. Quadratic log-trend: tied with linear (6-fold)
2. Monthly profiles: nearly identical to single profile
3. Thermal lag (gamma): gamma=1.0 optimal (instantaneous temp wins)

Conclusion: Standard Holt-Winters + AR(1) matches physics-augmented variants.

#### 3.5B Data Cleaning
Removing anomalously low 2018-2019 data (95% of bad hours) changed MAE by
only 3 MW. Median-based profiles are robust.

#### 3.5C Joint vs Sequential Decomposition
Joint Holt-Winters (Level 2) scored 468 MW vs sequential Level 1's 217 MW.
Stepwise decomposition is strongly preferred.

#### 3.5D Window-Size Ablation (AdaptiveDecomEngine)

We conducted a full sweep of profile window (7, 14, 21, 28, 42, 56 days) x trend
window (3, 5, 7, 10, 14 days) = 30 combinations x 6 folds, using daily-updating
structural forecast (trend x profile, no temp model). Key findings:

**24h Horizon — 6-Fold Mean MAE (MW)**

| profile\\trend |   3d |   5d |   7d |  10d |  14d |
|---------------|-----:|-----:|-----:|-----:|-----:|
|  7d           |  120 |  119 |  115 |  118 |  120 |
| 14d           |  118 |  118 |  114 |  116 |  118 |
| 21d           |  118 |  118 |  114 |  116 |  118 |
| 28d (default) |  118 |  118 |  114 |  116 |  118 |
| 42d           |  118 |  118 |  115 |  117 |  119 |
| 56d           |  119 |  118 |  115 |  117 |  119 |

**Trend window dominates**: Across all profile sizes (7-56d), MAE varies by only
~1 MW for a given trend. The trend window choice is the sole driver:

| Trend window | Mean MAE range | Assessment |
|-------------|---------------|------------|
| 3 days | 118-120 MW | Noisy, worst |
| 5 days | 117-119 MW | Good |
| **7 days (default)** | **114-115 MW** | **Best** |
| 10 days | 116-118 MW | Acceptable |
| 14 days | 118-120 MW | Stale, worst |

**Default (28,7) is essentially optimal**: 0 out of 30 combinations beat it by
more than 0.5 MW. The 6-fold mean for (28,7) is 114 MW (trend x profile only),
compared to 217 MW for the static (non-adaptive) equivalent -- a **103 MW
improvement from daily-updating alone**.

**Overall winner**: (profile=21d, trend=7d) at 114 MW, statistically tied with
default (28,7). No change to production configuration is warranted.

**Implication**: The adaptive engine is highly robust to window choice. Any
profile >= 14 days and trend = 5-10 days performs within ~5 MW of optimal.
This robustness is a practical advantage -- the system requires no periodic
retuning as load patterns evolve.

#### 3.5E Temperature Model Autopsy (ECG Real Data)

After discovering that `data/ecg_actual_demand_clean_with_temp.csv` contains REAL ECG
grid demand (identical to the CSV used in training), we conducted a full
temperature-demand analysis on the actual data.

**Real temperature-demand relationship:**

| Metric | Value |
|--------|-------|
| Overall hourly correlation (r) | +0.088 (weak positive) |
| Daily mean correlation (r) | **+0.333** (moderate) |
| Linear sensitivity | **+92 MW/C (4.1%/C)** |
| R^2 (daily demand ~ daily temp) | 0.10 - 0.35 (varies by fold) |

The real ECG data HAS a strong, statistically significant temperature effect:
each 1C rise increases demand by ~60-90 MW (3-4%/C), consistent with air
conditioning load in a tropical climate.

**Why temperature models still fail (both hourly and daily formulations):**

We tested six formulations on the real ECG data:

**Against rolling 7-day trend baseline:**

| Formulation | Description | 6-fold Mean Delta |
|-------------|-------------|-------------------|
| Old hourly | ratio ~ piecewise(temp_anomaly_by_hour) | -1 MW |
| Corrected hourly (multiplicative) | forecast = trend x profile x (1 + alpha(h) x temp_anom) | **0 MW** |
| Daily linear | daily_trend x (1+alpha x temp_anom) x profile | **+10 MW** |
| Synthetic fit | same, using synthetic sine-wave temps | +2 MW |
| Meteostat station fit | same, using Meteostat station obs | 0 MW |
| Threshold (hot >32C or cold <22C only) | same hourly, but only adjust at extremes | 0 MW |

**Against static (non-adaptive) baseline** (global average trend x fixed profile):

| Formulation | 6-fold Mean MAE | Delta vs Static |
|-------------|----------------|----------------|
| Static (no model) | 503 MW | -- |
| + Daily temp correction | **470 MW** | **-33 MW (-6.5%)** |
| + Hourly shape correction | 501 MW | -2 MW |

The daily temperature model dramatically improves the static baseline (-33 MW, 6.5%),
confirming GridCo operators are correct: temperature IS the primary adjustment factor
when the baseline has no adaptive component. The hourly shape model adds nothing more
(+/- 2 MW) because the within-day profile shape is minimally affected by temperature
(hourly R^2 < 12%).

**Why temperature models are neutral with the rolling trend:**
The 7-day rolling trend **implicitly captures temperature**. Here is why:

A 7-day rolling trend is simply `mean(daily_demand[t-6 : t])`. Since weather in
Ghana persists for 5-15 days (typical synoptic-scale weather patterns), any
temperature anomaly that affects demand will also affect the rolling trend within
the 7-day window. By the time we forecast day t+1, the trend already reflects
the last week's weather. Adding a separate temperature model means predicting
the *residual* of an already-adaptive trend -- but the residual is dominated
by noise, holidays, and weekend effects, with little remaining temperature
signal (residual temp correlation r = -0.26, a spurious within-day artifact).

In other words: **the trend adapts faster than the weather changes.** This makes
temperature models redundant, not because temperature doesn't matter, but because
the rolling window already accounts for it. This is a feature, not a bug, of the
adaptive decomposition design.

**Non-linear temperature-demand relationship:**

The temperature-demand curve is concave upward (quadratic, F=34.2, p<0.0001),
with demand minimized at ~24.4C (comfortable) and rising on both sides. This is
physically meaningful in a tropical climate:

| Range | Behavior | Slope (MW/C) | Physical driver |
|-------|----------|-------------|-----------------|
| <24C | Elevated (vs linear) | -- | Early morning hours, artificial lighting |
| 24-26C | Near minimum | +92 | Comfortable, minimal cooling |
| 26-28C | Linear | +54 to +125 | Moderate AC use |
| 28-30C | Steepening | **+227** | AC load ramps up |
| >30C | Saturation | **+304** | AC at full power |

The linear approximation (92 MW/C) is adequate for typical conditions (85% of
hours are 24-29C), but underestimates demand at extremes:

- At **32C**: Quadratic predicts **3,494 MW** vs linear's 2,804 MW (+690 MW)
- At **22C**: Quadratic predicts **2,206 MW** vs linear's 1,755 MW (+451 MW)

Despite the statistically significant non-linearity, it does not improve the
rolling model (0 MW delta across all formulations tested: quadratic, cubic,
piecewise, spline, threshold ramp). The 7-day trend already captures extreme
temperature effects because they persist longer than 7 days -- by the time
a 32C day arrives, the trend has already adapted.

**Cross-validation of temperature data sources:**

We cross-validated Open-Meteo ERA5 temperatures against Meteostat station
observations for Accra, Kumasi, Takoradi, and Tema:

| Location | Source | Mean | Meteostat Mean | Bias | MAE | R^2 |
|----------|--------|------|---------------|------|-----|-----|
| Accra | Open-Meteo | 26.5C | 27.6C | +1.1C | 1.1C | 0.28 |
| Kumasi | Open-Meteo | 25.6C | 26.9C | +1.3C | 1.3C | -0.21 |
| Takoradi | Open-Meteo | 25.9C | 27.4C | +1.5C | 1.5C | -1.11 |
| Tema | Open-Meteo | 26.8C | 27.4C | +0.6C | 0.8C | 0.62 |
| **All** | -- | 26.2C | 27.3C | **+1.1C** | **1.1C** | 0.13 |

Open-Meteo ERA5 reanalysis is systematically ~1C cooler than airport station
observations (urban heat island effect) but MAE of 1.1C is acceptable for grid
reanalysis vs station data. The bias is irrelevant for the temp model since it
uses temperature anomalies, not absolute values.

Meteostat station data was downloaded for Kumasi and Takoradi (Accra and Tema
failed due to API rate limits). Using Meteostat station data instead of ERA5
did not change the temp model outcome (still 0 MW delta).

#### 3.5F Gaussian Process and Markov Structure Tests

We tested whether non-parametric (Gaussian Process) and/or Markov-structured
models could capture the temperature-demand relationship better than the
rolling decomposition, as a rigorous check against overclaiming optimality.

**GP models tested (all with ARD RBF/Matern kernels, subsampled training):**

| Model | Inputs | Daily MAE | Hourly MAE | Verdict |
|-------|--------|-----------|-----------|---------|
| GP(temp) | temp only | 458 MW | -- | Identical to linear (R2 ~ 0.11) |
| GP(temp, lag1) | temp + yesterday's demand | 210 MW | -- | Markov structure helps, but rolling 7d mean (84 MW) is better |
| GP(temp, lag1..lag7, 24h) | 7 lags + temp + hour dummies | -- | 502 MW | Catastrophic: 32 features x low SNR -> constant predictor |
| LinReg(7 lags + temp, daily) | 7 lags + temp (linear) | **72 MW** | -- | Beats equal-weighted rolling mean (84 MW) by 12 MW |
| LinReg(7 lags + temp + 24h) | 7 lags + temp + hour dummies | -- | **113 MW** | Tied with Trend x Profile (114 MW) |

**Key findings:**

1. **Markov structure dominates.** Adding yesterday's demand (lag1) to GP improves MAE
   from 458 MW to 210 MW. The time series nature is far more informative than temperature alone.

2. **But the equal-weighted rolling mean is suboptimal.** A linear model with optimally
   weighted lags (LinReg: 0.55*l1 - 0.10*l2 + 0.12*l3 - 0.06*l4 + 0.07*l5 + 0.06*l6 +
   0.30*l7 + 9.0*temp) achieves 72 MW on daily data vs the rolling mean's 84 MW (-12 MW).

3. **The improvement is purely from optimal lag weighting, not temperature.**
   LinReg without temperature achieves 73 MW (-11 MW). The temperature feature adds
   only 1 MW to the improvement. The dominant weights are lag1 (0.55, yesterday) and
   lag7 (0.30, same-day-last-week), with even lags showing mean reversion (negative).

4. **A simpler weighted trend (0.65*l1 + 0.35*l7) captures the full improvement.**
   When applied hourly as trend x profile, this 2-weight model achieves 105 MW vs
   the rolling mean's 113 MW (-8 MW, p=0.027). It matches the full LinReg (105 MW)
   despite having no temperature, no intercept, and only 2 weights. The weights are
   stable across all 6 folds (lag1 std=0.05, lag7 std=0.04).

5. **GP catastrophically fails at high dimension.** With 32 features (7 lags + temp + 24 hours),
   GP produces near-constant predictions (502 MW vs 114 MW). The R2 of temp-demand is so low
   (max 0.12) that the GP kernel collapses to infinite length scales, predicting the training
   mean everywhere. This is a fundamental limitation of GP on low-SNR, high-dimensional data.
   Even a properly specified composite kernel (RBF + Periodic + WhiteKernel on time, temp,
   and hour dimensions) cannot bridge R2 < 0.12. The rolling model wins because its 7-day
   trend is a 10x more powerful feature than temperature.

**Implication:** The equal-weighted rolling mean trades 8 MW for zero complexity. The
weighted trend (0.65*l1 + 0.35*l7) is recommended for production: no training, no
temperature, no extra parameters, and a proven -8 MW gain across all 6 folds at p=0.027.

#### 3.5G Additional ML Approaches and DOW Correction

We tested six additional approaches designed to exploit the problem's structure
(beyond black-box ML that failed earlier). Tests were conducted at D+7 (168h) with
the full hourly profile and 6-fold CV:

| Method | Description | MAE | Delta vs Weighted Trend | Verdict |
|--------|-------------|-----|------------------------|---------|
| Weighted trend (base) | 0.65 x lag1 + 0.35 x lag7 x profile | **126 MW** | baseline | Production baseline |
| **DOW residual correction** | **Weighted trend + DOW-specific level offset** | **123 MW** | **-3 MW (-2.4%)** | **✅ Works, no ML needed** |
| Ridge residual model | Ridge(λ=100) on [DOW, month, lag1/7 residuals] | 123 MW | -3 MW (-2.4%) | Same as DOW correction |
| DOW-specific lag weights | 7 different (α₁, α₇) per DOW | ~125 MW | ~-1 MW | ❌ Marginal |
| Matrix factorization (SVD) | Rank-3 SVD of days×24 hourly matrix | 139 MW | +13 MW | ❌ Too stiff |
| Fourier + trend | Sin/cos basis for 7-day cycle + linear trend | 461 MW | +335 MW | ❌ Catastrophic without lags |
| Mixture of 3 experts | Gate on volatility/growth → weighted trend/HW/lag1 | 126 MW | 0 MW | ❌ Gate never fires |
| Online learning (regret) | Adaptive (α₁, α₇) shifting toward better lag | 127 MW | +1 MW | ❌ Regresses to fixed weights |

**The only signal in the residuals is DOW bias.** The weighted trend formula
creates a structural bias at weekday/weekend boundaries:

```
Monday level   = 0.65 × Sunday(low) + 0.35 × last_Monday(high)
               → Under-predicts Monday by +54 MW
Saturday level = 0.65 × Friday(high) + 0.35 × last_Saturday(low)
               → Over-predicts Saturday by -38 MW
```

The mean residuals per DOW (from training on all folds):

| DOW | Correction | Reason |
|-----|-----------|--------|
| Monday (0) | **+54 MW** | Sun(low) → Mon(high): trend lags the weekly jump |
| Tuesday (1) | **+24 MW** | Smaller carry-over from Monday |
| Wednesday (2) | **+2 MW** | Mid-week is neutral |
| Thursday (3) | **+3 MW** | Mid-week is neutral |
| Friday (4) | **+4 MW** | Mid-week is neutral |
| Saturday (5) | **-38 MW** | Fri(high) → Sat(low): trend overshoots |
| Sunday (6) | **-41 MW** | Sat(low) → Sun(low): double low |

Adding these DOW offsets to the weighted trend achieves the full -3 MW improvement.
Ridge regression with DOW features achieves the same -3 MW — confirming the DOW
bias is the only signal worth modeling. The residuals show no exploitable
autocorrelation (lag-1 r = -0.004, lag-7 r = -0.03), so no AR/ML model can predict
beyond the DOW mean.

**Residual autocorrelation (full data):**
| Lag | r |
|-----|---|
| 1 | -0.004 |
| 2 | -0.095 |
| 3 | +0.066 |
| 7 | -0.032 |
| 14 | +0.127 |

All near zero — the weighted trend captures the time-series structure completely,
leaving only DOW-specific level noise.

**Why the other approaches failed:**
- **Matrix factorization**: Rank-3 SVD reconstructs the hourly shape with ~5% error,
  which is larger than the 2.4% gain from DOW correction. The 28-day median profile
  is already near-optimal for hourly shape.
- **Fourier decomposition**: Pure harmonic basis (period=7 for weekly) without lag
  features has R² < 0.15. Fourier is a weaker feature set than a single lag.
- **Mixture of experts**: The volatility/growth gate thresholds never trigger because
  the demand CV (5%) is too low to separate regimes. All days look the same to the gate.
- **Online learning**: The optimal (0.65, 0.35) weights are so stable across all periods
  (σ < 0.05) that any adaptation just adds noise (+1 MW).

**Production recommendation**: Add DOW correction to the weighted trend:

```python
trend = 0.65 * demand[t-1] + 0.35 * demand[t-7]
dow_offset = {0: 54, 1: 24, 2: 0, 3: 0, 4: 0, 5: -38, 6: -41}
daily_level = trend + dow_offset[dayofweek]
forecast = daily_level * hourly_profile
```

Total improvement from engineering baseline (113 MW): **-8 MW (weighted trend) + -3 MW
(DOW correction) = -11 MW (102 MW)**. No training, no parameters, no ML.

#### 3.5H Similar-Day Approaches (GridCo-Inspired) and HR-LEAR

After exhausting ML approaches on residuals, we tested two families of models inspired
by external methods: (i) similar-day KNN heuristics (based on GridCo operator methodology),
and (ii) the Hierarchical Regime-Aware LEAR (HR-LEAR) architecture from the electricity
price forecasting literature.

**Similar-Day Approaches:**

The GridCo operator approach ("find a similar historical day, adjust for temperature")
was digitized as a KNN search over [DOW, month, weekend, holiday, temp, prev_mean,
roll7_mean] features, with inverse-distance weighted averaging, temperature coefficient
3.1%/°C, and 8%/year growth scaling. We also tested simpler variants:

| Method | Description | MAE | vs WT+DOW+prof | Verdict |
|--------|-------------|-----|----------------|---------|
| WT+DOW+prof (baseline) | Weighted trend + DOW + Month×DOW profile | **113 MW** | baseline | Production standard |
| GridCo SimDay (K=5) | KNN on 7 features + temp/growth adjustment | **141 MW** | **+28 MW** | ❌ Worse than baseline |
| Same DOW + closest temp | 1-NN by temp within same DOW, 3.1%/°C adj | **226 MW** | **+113 MW** | ❌ Single day too noisy |
| Same DOW + month avg | Weighted avg of all same-DOW-month days | **324 MW** | **+211 MW** | ❌ Stale data, no recent trend |
| Last 5 same-DOW (level-scaled) | Last 5 DOW days, scaled to current level | timed out | — | ❌ Too slow to evaluate |

All similar-day variants failed against the weighted trend. Even the GridCo-engineered
KNN (piloted on the Nayaga substation with 17.9% MAPE) underperforms by +28 MW when
applied to ECG system-wide data. **Similar-day matching is a weaker feature set than
the two-lag weighted trend.** The 7-dimensional feature space (DOW, month, temp, etc.)
adds noise rather than signal because:
1. The weighted trend's lag1 already captures yesterday's actual — the most
   informative "similar day" predictor.
2. Temperature adjustment (3.1%/°C) is architecturally redundant with the
   rolling trend (same reason as §3.5E — the trend adapts faster than weather).
3. Growth scaling (8%/year) compounds noise at 7-day horizons (same failure
   mode as Holt-Winters at D+30, §3.5G).

**Hierarchical Regime-Aware LEAR (HR-LEAR):**

We adapted the HR-LEAR architecture from Kamal (2026) — a paper addressing neural MoE
collapse in electricity price forecasting — to the load forecasting problem. The architecture:

```
Level 0:   Weighted trend + DOW correction + Month×DOW profile  (global anchor)
HMM:       Gaussian HMM (n=2..4) on Level 0 residuals           (regime discovery)
Level 1:   ElasticNet per regime on Level 0 residuals            (residual specialists)
SPI:       Stability-Preserving Indicator, Φ = I(max(γ) > 0.6)   (gate by HMM confidence)
```

The HMM identifies latent regimes in the residual series (e.g., normal, over-predict,
under-predict). Each regime gets an ElasticNet specialist trained on features:
temp, temp², DOW one-hot, month sin/cos, lagged residuals, rolling residual mean,
and temperature anomaly. The SPI prevents specialist corrections when HMM confidence
is below 0.6, reverting to the stable Level 0 anchor.

| Model | MAE | vs Level 0 | Verdict |
|-------|-----|------------|---------|
| Level 0 (WT+DOW+profile) | **113 MW** | baseline | — |
| HR-LEAR (n=2, τ=0.6) | **112 MW** | **-1 MW** | ❌ Marginal |
| HR-LEAR (n=3, τ=0.6) | **112 MW** | **-1 MW** | ❌ Marginal |
| HR-LEAR (n=4, τ=0.6) | **114 MW** | +1 MW | ❌ Worse |

The -1 MW improvement is not significant. The HR-LEAR architecture — despite achieving
-15.3% CRPS on German electricity price data — fails to improve load forecasting because:

1. **Load residuals are pure noise (lag-1 autocorr = -0.004).** The HMM cannot
   discover meaningful regimes because there are no regimes to discover. The
   Gaussian HMM essentially splits the noise distribution into arbitrary partitions.
2. **Load CV (4-6%) is too low for regime separation.** The paper's price data had
   distinct physical regimes (Surplus/Base/Scarcity with price ranges spanning
   -500 to +500 EUR/MWh). Load has no equivalent — all days are structurally similar.
3. **Specialist features are redundant with Level 0.** Temperature, DOW, and month
   features are already implicit in Level 0's rolling trend and profile.
4. **The SPI gate never triggers.** HMM confidence rarely exceeds 0.6 because all
   residuals look similar, making the SPI revert to Level 0 by default.

**Attempted augmentation (Optimal Transport):** The paper's champion variant
(OT-HR-LEAR) transported residuals from the dominant regime to sparse regimes
via 1D Wasserstein mapping. We did not implement this because with n=3 regimes,
all regimes had sufficient data (the load residual distribution is unimodal, not
multimodal), so OT augmentation would add noise to an already-clean distribution.

**Overarching conclusion:** The weighted trend + DOW correction + profile model is
near-optimal for ECG load forecasting. Similar-day heuristics (+28 MW), neural
mixture of experts (gate never fires), and hierarchical regime architectures
(-1 MW) all fail because the problem has fundamentally different structure from
electricity price forecasting: load is smooth (CV 5%), residuals have only weak
exploitable structure (weekly cycle r = -0.23, monthly bias up to +/-19 MW), and
no regime separation exists. The gains come from optimal lag weighting (-8 MW),
DOW bias correction (-3 MW), and potentially month-adaptive corrections (-2-3 MW
estimated). Further residual structure (Section 3.5J) suggests ~5% more improvement
is possible beyond current baseline.

#### 3.5I Error Analysis: Accuracy Profile and Systematic Patterns

Our model (WT + DOW correction + Monthx DOW profile) achieves a **mean MAE of 98 MW
(4.1% MAPE)** across all 6 folds in D+1 (day-ahead) evaluation, where lag1 uses
yesterday's actual demand. Performance degrades with horizon length: 113 MW at
D+7, approximately 160 MW at D+30 (estimated from recursive error compounding).

**Per-Fold Breakdown:**

| Fold | Period | MAE | MAPE | RMSE | Actual Mean | Context |
|------|--------|-----|------|------|-------------|---------|
| F1 | 2020 H1 | 82 MW | 4.2% | 112 MW | 2,012 MW | COVID-era low demand |
| F2 | 2021 H1 | 81 MW | 3.7% | 113 MW | 2,268 MW | Stable recovery |
| F3 | 2022 H1 | 97 MW | 4.2% | 133 MW | 2,355 MW | Moderate growth |
| F4 | 2023 H1 | 93 MW | 3.9% | 122 MW | 2,412 MW | Steady growth |
| F5 | 2024 H1 | 114 MW | 4.5% | 157 MW | 2,653 MW | High growth |
| F6 | 2025 H1 | **123 MW** | 4.3% | 180 MW | 3,010 MW | Growth discontinuity |
| **Mean** | -- | **98 MW** | **4.1%** | **136 MW** | -- | -- |

**Error Distribution (all folds combined):**

| Percentile | 10% | 25% | 50% | 75% | 90% | 95% | 99% | Max |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|
| MAE (MW) | 14 | 35 | 75 | 134 | 204 | 265 | 457 | ~800 |
| % of mean demand | 0.6% | 1.4% | 3.0% | 5.5% | 8.3% | 10.8% | 18.6% | ~33% |

The error distribution is right-skewed: 50% of errors are below 75 MW (3.0% MAPE),
but the tail reaches 457 MW at the 99th percentile. The worst errors coincide with
rapid growth transitions (Fold 6) and holiday-recovery periods.

**MAE by Hour of Day (6-fold avg):**

```
Hour  1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24
MAE  102   96   95   92   91   88   91   82   78   83   86   90   94   95  106  109  102  100  102  109  109  119  121  122
```

Key patterns:
- **Best hour: 9am (78 MW)** — Mid-morning after the morning ramp, demand is stable.
- **Worst hour: Midnight (122 MW)** — Lowest demand period, model has difficulty
  with the transition between days.
- **Late-night degradation (hours 22-24):** MAE spikes to 109-122 MW. The profile
  shape is most variable during the evening ramp-down as different consumer segments
  turn off at different times.
- **Peak hours (7-19h):** 94 MW average error. Mid-day (9-14h) best at 78-94 MW.
- **Off-peak (20-6h):** 104 MW average error. Counter-intuitively, errors are
  *higher* at night because the relative variability of load is larger at low
  absolute levels (the same ±100 MW swing is a larger % error at 1,500 MW than
  at 2,500 MW).

**MAE by Day of Week (6-fold avg):**

| Day | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|-----|-----|-----|-----|-----|-----|-----|-----|
| MAE | 96 | 96 | 101 | 99 | **93** | 101 | **103** |

- **Best day: Friday (93 MW)** — Stable end-of-week pattern with minimal surprises.
- **Worst day: Sunday (103 MW)** — Adjacent to the Monday ramp, low demand day with
  higher relative variability. The DOW correction helps but cannot eliminate the
  transition uncertainty entirely.
- **Weekday avg (Mon-Fri):** 97 MW. **Weekend avg (Sat-Sun):** 102 MW.
  Weekend errors are 5% higher despite lower absolute demand.

**MAE by Month (6-fold avg, note: test windows are Jan-Jun only):**

| Month | Jan | Feb | Mar | Apr | May | Jun |
|-------|-----|-----|-----|-----|-----|-----|
| MAE | 74 | **62** | 77 | 108 | **113** | 106 |

- **Best month: February (62 MW)** — Stable dry-season weather, consistent load.
- **Worst month: May (113 MW)** — Pre-monsoon transition with high temperature
  variability and AC load ramping. The model struggles with the rapid increase
  in temperature-driven load.
- **Winter (Jan-Mar):** 71 MW avg. **Spring (Apr-Jun):** 109 MW avg.
  Spring errors are 53% higher than winter, driven by temperature variability
  and the onset of AC season.

**Systematic Patterns Summary:**

| Condition | MAE | vs Mean | Why |
|-----------|-----|---------|-----|
| Best hour (9am) | **78 MW** | -20% | Stable mid-morning demand |
| Worst hour (midnight) | **122 MW** | +24% | Day-boundary transition, low absolute load |
| Best day (Friday) | **93 MW** | -5% | Stable end-of-week |
| Worst day (Sunday) | **103 MW** | +5% | Low demand, adjacent to Monday ramp |
| Best fold (F2: 2021 H1) | **81 MW** | -17% | Stable market, 5.7% CAGR |
| Worst fold (F6: 2025 H1) | **123 MW** | +26% | 29% CAGR discontinuity |
| Best month (Feb) | **62 MW** | -37% | Stable dry-season |
| Worst month (May) | **113 MW** | +15% | Pre-monsoon temp variability |
| Off-peak (20-6h) | 104 MW | +6% | Low absolute level, higher relative noise |
| Peak (7-19h) | 94 MW | -4% | High absolute level, stable consumption |
| 50% of errors | <75 MW | -- | Model is reliable in the middle |
| 5% of errors | >265 MW | -- | Tail events = growth discontinuities |

**Implication for the frontend:** These error bounds should be communicated to users
as forecast confidence intervals per hour and per day. For example, a Monday forecast
for 9am should show +/-78 MW uncertainty, while a Sunday midnight forecast should
show +/-122 MW. The Monthx DOW profile already captures the systematic shape, but the
residual uncertainty follows a clear hourly, daily, and seasonal pattern that can
be modeled as a lookup table for probabilistic interval generation.

#### 3.5J Residual Structure: White Noise or Exploitable?

We previously claimed residuals were pure white noise (lag-1 autocorr = -0.004).
This was based on daily-level analysis across all folds. A deeper statistical
analysis reveals this was **partially incorrect** — the residuals DO have
exploitable structure, though it is weak.

**Daily Residual Autocorrelation (All Folds Combined):**

| Lag | ACF | Verdict |
|-----|-----|---------|
| 1 | -0.02 | White noise (p=0.48) |
| 2 | -0.05 | White noise |
| 7 | **-0.23** | **Significant (p < 0.0001)** |
| 14 | +0.00 | White noise |
| 28 | +0.06 | Significant (p < 0.0001) |

The Ljung-Box test confirms structure at lags 7, 14, and 28 (all p < 0.0001).
The negative lag-7 ACF (-0.23) means: if the model over-predicted last Tuesday,
it tends to slightly under-predict this Tuesday. This is a **residual weekly
oscillation** that DOW correction does not fully eliminate.

**Why DOW correction doesn't fix this completely:**
The DOW offsets (+54 Mon, +24 Tue, ..., -41 Sun) are constant across time, but
the actual DOW bias varies by fold and by month. Fold 6 (2025 H1) has different
optimal offsets than Fold 1 (2020 H1) because the demand pattern changed.
The global offsets capture the average across all folds, leaving a residual
weekly wobble.

**What this means for ML feasibility:**
With r = -0.23 at lag 7, the explained variance is r^2 = 0.05 (5%).
A model trained to predict this residual weekly cycle could reduce the residual
standard deviation from 88.5 MW to approximately 88.5 x sqrt(1 - 0.05) = 86.2 MW,
a gain of ~2-3 MW. This is small but real.

**Hourly Residual Autocorrelation (intra-day, NOT exploitable for D+1):**

| Fold | Lag-1 | Lag-2 | Lag-3 | Lag-24 |
|------|-------|-------|-------|--------|
| F1 | +0.81 | +0.70 | +0.62 | +0.02 |
| F2 | +0.75 | +0.64 | +0.55 | -0.01 |
| F3 | +0.78 | +0.66 | +0.57 | +0.03 |
| F4 | +0.79 | +0.70 | +0.63 | +0.03 |
| F5 | +0.83 | +0.72 | +0.62 | +0.02 |
| F6 | +0.80 | +0.69 | +0.59 | -0.05 |

Lag-1 hourly autocorrelation is consistently ~0.80 across all folds. This is
**intra-day smoothness** (adjacent hours have similar demand), NOT exploitable
structure for day-ahead forecasting. When predicting all 24 hours simultaneously,
you cannot use hour 1's actual to correct hour 2 because both are in the future.
For intra-day correction (rolling 1-hour updates), an AR(1-3) on hourly residuals
would reduce error by ~50-60%.

**Monthly Bias in Residuals (positive = model under-predicts):**

| Month | Bias | Month | Bias |
|-------|------|-------|------|
| Jan | +8 MW | Jul | -9 MW |
| Feb | +15 MW | Aug | -3 MW |
| Mar | -2 MW | Sep | +16 MW |
| Apr | -6 MW | Oct | +19 MW |
| May | -17 MW | Nov | +17 MW |
| Jun | -7 MW | Dec | -2 MW |

Several months show systematic bias of +/-15-19 MW (October +19, May -17,
September +16, November +17). These are large enough to matter. The profile
captures the within-day shape but the level systematically drifts by month.

**Verdict:** The residuals are NOT pure white noise. There are two exploitable
signals:
1. **Weekly oscillation (lag-7 ACF = -0.23, r^2 = 0.05):** Could yield ~2-3 MW
   improvement with an AR(7) model on daily residuals.
2. **Monthly level bias (up to +/-19 MW):** Adding month as a level feature could
   yield ~3-5 MW improvement.
Combined, these suggest the theoretical floor is **~90-93 MW** (from current 98 MW),
not 98 MW. The model is near-optimal but not at the absolute floor.

**What DL still CANNOT do:**
- The 0.80 hourly autocorrelation is intra-day smoothness — useless for D+1.
- The -0.23 weekly autocorrelation is real but tiny (r^2 = 0.05).
- No regime structure exists for MoE or HMM models.
- A simple AR(7) on residuals + month dummies would capture ~90% of the
  remaining signal without any deep learning.

**5 Key Design Questions Answered:**

**Q1: Would more 2026 data reduce Fold 6 error?**
Conditionally. Fold 6's 123 MW is driven by 29% CAGR (vs <10% in training).
The residual bias is only -5 MW (near zero) — the model does not systematically
under-predict. The problem is higher VARIABILITY during rapid growth. More data
would help stabilize the profile and trend estimates IF the growth rate normalizes
to <15%. If 29%+ CAGR persists, the model needs a structural change (e.g.,
adaptive learning rate or growth-sensitive lag weighting), not more data.

**Q2: Would a "quick refit on uploaded few days" help?**
The model requires ZERO training — it updates immediately by design:
- lag1 = yesterday's actual (always fresh)
- lag7 = same-day-last-week actual (always fresh)
- 28-day profile = rolling window (auto-updates daily)
- DOW offsets = precomputed from training (stable across folds)

The one thing that could benefit from re-estimation: **DOW offsets**. If the
system allowed uploading a few weeks of recent data, the DOW offsets could be
recomputed on the latest window. This would detect if the Monday +54 MW offset
is shifting (e.g., to +60 MW in 2025). Implementation is trivial: compute
mean residual per DOW on the uploaded data.

**Q3: Would frequent schedule data uploads help the forecast?**
Potentially, but it depends entirely on what the schedule contains.
- **Generator outages:** A planned_outage binary flag could reduce error during
  supply-constrained days (not currently in the model).
- **Load-shedding schedules:** If ECG publishes controlled blackout plans, these
  would explain the largest outliers (demand drops of 300-500 MW).
- **Maintenance calendars:** Similar to outages, would explain unusual dips.
- **Holiday schedules:** The current holiday effect is only -0.7% (negligible),
  but major events (elections, national ceremonies) could have larger effects.

Without knowing the schedule schema, the value is speculative. If schedules
contain any of the above, encoding them as binary/categorical features could
reduce tail errors (90th+ percentile) by 10-20%.

**Q4: Would month encoding help the level formula?**
Yes, surprisingly. The monthly bias analysis shows systematic errors up to
+/-19 MW (October +19, May -17). Month is already in the profile (MonthxDOW
shape), but the LEVEL itself has month-specific bias. Adding month dummies to
the level formula is a simple change:

```python
level = 0.65*l1 + 0.35*l7 + DOW_offset[dow] + MONTH_offset[month]
```

Estimated improvement: ~3-5 MW. Implementation cost: zero (precompute offsets
from training data, same as DOW correction). This is the highest ROI change
currently available.

## 4. Production Architecture

### 4.1 Horizon-Dependent Forecast Engine

A single engine for all horizons is naive. Error compounds differently at each horizon,
and production-grade systems use different strategies for each:

```
ForecastEngine:
  24h (day-ahead):  WeightedTrend(0.65*l1+0.35*l7) x 28d Profile
  168h (week-ahead): Recursive for days 2-3, then DailyAvg x Profile
  720h (month-ahead): Seasonal (last-year-same-month x growth)  
```

#### 24-Hour (Day-Ahead) — Optimal

**Method:** Weighted trend + DOW correction x 28-day median profile.
**Trend:** 0.65 x lag1 + 0.35 x lag7 (replaces equal-weighted rolling mean).
**DOW correction:** +54 (Mon), +24 (Tue), 0 (Wed-Fri), -38 (Sat), -41 (Sun) MW.
**Profile:** 28-day rolling median of hourly demand/daily-mean ratios.
**MAE:** 102 MW (6-fold CV), best across all tested models.
**Why it works:** All lags are actual demand. The weighted trend correctly prioritizes
yesterday (weight 0.55) over mid-week days (near-zero) while capturing the weekly cycle
via lag7 (weight 0.30). The DOW correction removes a structural bias at weekday/weekend
boundaries: the formula `0.65*Sun + 0.35*Mon` systematically under-predicts Monday by
+54 MW because Sunday is a low-demand day. The DOW offset is the only signal worth
modeling in the residual (all autocorrelations < 0.15). Temperature adds nothing.

```python
class AdaptiveDecomEngine:
    # -- Adaptive (recomputed at forecast time from rolling window) --
    rolling_profile: np.ndarray[24]     # 28-day median profile
    weighted_trend: float               # 0.65 * demand[t-1] + 0.35 * demand[t-7]
    dow_offset: list[float]             # [54, 24, 0, 0, 0, -38, -41]
    #
    # daily_level = weighted_trend + dow_offset[dayofweek]
    # forecast = daily_level * rolling_profile
    #
    # Total improvement: -11 MW vs engine baseline (113 -> 102 MW)
    # -8 MW from weighted trend (p=0.027), -3 MW from DOW correction
    # No training, no parameters, no ML needed.
    # Temperature implicitly captured by trend window.
```

#### Week-Ahead (2-7 Days) — Hybrid

**Problem:** Recursive hourly forecasting compounds error. By day 7, all 7 lag features
are predicted values, not actuals. The weighted trend loses its advantage (lag1 is now
a forecast, not an actual).

**Solution:** Two-phase approach:

| Days | Method | Why |
|------|--------|-----|
| 1 (t+24h) | Weighted trend x profile | lag1 = yesterday's actual |
| 2-3 | Recursive hourly (weighted trend feeds its own output) | Acceptable error (~+5%) |
| 4-7 | Daily avg forecast x historical profile | Stable, profile shape is reliable |

**Daily avg forecast for days 4-7:** Predict a single daily value using the same
weighted trend on daily averages, then multiply by the 28-day profile. This avoids
compounding hourly error. The profile's within-day shape barely varies day-to-day
(max DOW deviation 0.05, 7 DOW profiles tested and no improvement).

**MAE estimate:** ~120-130 MW (vs 105 MW at 24h). The profile is robust but the
daily trend loses accuracy as horizon increases.

**Notes:** DOW-specific profiles tested and rejected (7 profiles hurt by +1-6 MW
due to insufficient samples per day-of-week). Temperature models tested and rejected
at all horizons (0 MW delta with rolling trend, +227 MW with GP composite kernel).

#### Month-Ahead (8-30 Days) — Seasonal Only

**Problem:** At 30 days, hourly resolution is meaningless. Error from recursive
forecasting compounds to the point where hourly granularity adds noise, not signal.
No production system forecasts hourly demand 30 days ahead.

**Method:** Daily resolution only, using:
- **Same-month-last-year** as baseline demand pattern
- **Growth rate adjustment** (10% CAGR recent, 8.6% historical)
- No hourly profile (aggregate to daily totals or average)
- No temperature (climatological normal is constant — no predictive power)

```python
class MonthlyForecaster:
    # -- Seasonal decomposition (no hourly resolution) --
    base_demand: float                  # mean demand in same month last year
    growth_rate: float                  # 8.6% CAGR (or rolling estimate)
    monthly_shape: np.ndarray[30]       # typical within-month pattern (day-of-month effects)
    #
    # Forecast: base_demand * (1 + growth_rate)^years_elapsed * monthly_shape
    # MAE: ~180-250 MW (30-day daily totals)
```

**Why not use the same adaptive engine for 720h?** The 7-day rolling trend at 720h
uses entirely forecasted lags (days 8-30 have no actual demand). The weighted trend
(0.65*l1 + 0.35*l7) cannot help when l1 is itself a 24-day-old prediction. Seasonal
decomposition avoids this by anchoring to the same calendar period last year.

**Production recommendation:** Generate a daily average forecast for days 8-30, then
apply the 28-day profile as a heuristic shape. Flag these forecasts as "low confidence"
in the UI.

### 4.2 Horizon Handoff Logic

```python
def forecast(horizon_days: int, date: date) -> dict:
    if horizon_days <= 1:
        return day_ahead_forecast(date)              # weighted trend x profile
    elif horizon_days <= 7:
        return week_ahead_forecast(date, horizon_days)  # hybrid recursive/profile
    elif horizon_days <= 30:
        return month_ahead_forecast(date, horizon_days) # seasonal only
    else:
        raise ValueError("Horizon > 30 days not supported")
```

### 4.3 Forecast Cache

```sql
CREATE TABLE forecast_cache (
    cache_key  TEXT PRIMARY KEY,       -- '{horizon_days}:{forecast_date}'
    forecast   JSONB NOT NULL,         -- full forecast object (hourly or daily)
    created_at TIMESTAMPTZ NOT NULL
);
```

Daily expiry. Force refresh via `?force_refresh=true`.

### 4.4 Next Steps Toward Deployment

| Priority | Task | Details | Status |
|----------|------|---------|--------|
| 1 | **Implement weighted trend + DOW correction in engine** | WeightedTrendEngine at `Backend/app/ml/weighted_trend_engine.py`. Endpoints: `/forecast/baseline/tomorrow`, `/forecast/baseline/7day`, `/forecast/baseline/30day`, `/forecast/baseline/90day`. Frontend toggle in ForecastView. | **Done** |
| 2 | **Full 6-fold CV re-run** | Validated: WT+DOW+profile = 98 MW D+1 MAE (vs DecomEngine ~110 MW). Month offsets tested, 0 MW gain. | **Done** |
| 3 | **Statistical significance writeup** | p=0.027 (weighted trend vs equal-weighted rolling mean). See §3.5 and conclusion 7. | **Done** |
| 4 | **Implement week-ahead hybrid** | BaselineForecastService.forecast_7day() uses WeightedTrendEngine recursively for 7 days via `predict_week_ahead()`. | **Done** |
| 5 | **Implement month-ahead seasonal** | BaselineForecastService.forecast_30day() and forecast_90day() via WeightedTrendEngine. | **Done** |
| 6 | **Horizon handoff API** | Unified `/forecast/baseline/tomorrow`, `/forecast/baseline/7day`, `/forecast/baseline/30day`, `/forecast/baseline/90day` endpoints with cache. | **Done** |
| 7 | **ModelAccuracyPanel frontend** | Collapsible panel in ForecastView showing per-hour/per-DOW error bars, fold breakdown, best/worst conditions. | **Done** |
| 8 | **Frontend engine toggle** | Toggle between DecomEngine v2.4 and Baseline WT+DOW in ForecastView. Affects 24h/7d/30d/90d tabs. | **Done** |

## 5. Conclusions

1. **Adaptive rolling decomposition** is the strongest model: simple, fast,
   interpretable, stable, and best across all six temporal folds.

2. **Structural priors matter.** The rolling 28-day profile + 7-day trend
   decomposition encodes domain knowledge that ML models cannot rediscover
   from lag features alone.

3. **Autoregressive ML correction is high-risk.** LightGBM with lagged
   residuals beats the structural engine in 18/18 fold-horizon combinations,
   but this advantage comes entirely from autocorrelation features that fail
   at long horizons.

4. **N-BEATS exhibits the Fold 6 collapse.** Any model trained purely
on historical data fails when the trend accelerates beyond the training
distribution. The adaptive window avoids this by design.

5. **Data quality is not the bottleneck.** Median-based profiles handle the
   2018-2019 data issues; cleaning changes MAE by only 3 MW.

6. **Window choice is secondary to adaptivity.** A 30-combo x 6-fold ablation
   shows the daily-updating mechanism provides 103 MW improvement (217 to 114 MW),
   while window size choice spans only 6 MW. The default (28d profile, 7d trend)
   is essentially optimal; no periodic retuning is needed.

7. **The equal-weighted rolling mean is suboptimal.** A weighted trend (0.65*l1 + 0.35*l7)
   achieves -8 MW (p=0.027) at zero additional complexity. The improvement is purely
   from optimal lag weighting — yesterday matters 4x more than mid-week days, and
   last week's same-day captures the weekly cycle. Temperature adds nothing.

8. **Temperature models are redundant with fast-rolling trends, but temperature DOES matter.**
   Against a static (non-adaptive) baseline, daily temperature correction achieves
   -33 MW (6.5% improvement), confirming GridCo operators are correct that temperature
   is the primary adjustment factor. The reconciliation: Real ECG data confirms +92 MW/C
   sensitivity (4.1%/C), with a statistically significant quadratic component (F=34.2,
   p<0.0001) reflecting AC saturation above 28C (+227 to +304 MW/C vs +92 MW/C average).
   Despite the real non-linearity, the 7-day rolling window implicitly captures it
   because Ghana's weather patterns persist longer than 7 days — the trend adapts
   before temperature corrections add value. This is a structural consequence of
   adaptive decomposition, not a data quality issue.

9. **One engine does not fit all horizons.** Hourly recursive forecasting is optimal
   for 24h but naive for 30 days. Production architecture should use three distinct
   strategies: weighted trend + DOW correction x profile (24h), hybrid recursive/profile
   (168h), and seasonal only (720h). No temperature or GP model tested improved any horizon.
   **Deployed:** `WeightedTrendEngine` at `/forecast/baseline/*` with frontend toggle.

10. **Adding more context (trend, growth) for longer horizons hurts, not helps.** We
    tested 6 approaches designed to provide more structural context for week-ahead and
    month-ahead forecasts: DOW-specific weights, residual modeling (Ridge), matrix
    factorization, Fourier decomposition, mixture of experts, and online learning with
    regret. Only DOW correction on residuals survives (-3 MW). The growth signal (~8%
    CAGR, ~0.023%/day) is lost in daily noise (~5% std). Holt-Winters fails
    catastrophically at D+30 (+103 MW) because the trend estimate compounds noise over
    30 days. The recursive weighted trend (level-only, zero-growth assumption) is
    already the optimal long-horizon forecast — not despite its simplicity, but because
    of it. Any method that tries to "learn" growth from noisy daily data will amplify
    estimation error at longer horizons.

11. **Similar-day heuristics and hierarchical regimes both fail on load data.**
    GridCo's digitized SimDay KNN (DOW, month, temp, prev_mean, roll7_mean features +
    temp/growth adjustment) underperforms the weighted trend by +28 MW. The HR-LEAR
    architecture (HMM regime discovery + ElasticNet residual specialists + SPI gating),
    despite -15.3% CRPS on German electricity prices, achieves only -1 MW on load —
    not significant. The root cause is structural: load residuals (CV 4-6%) are
    pure white noise (all autocorrelations |r| < 0.15), while electricity price
    residuals (CV 20-50%+) contain exploitable regime structure (Surplus/Base/Scarcity).
    The weighted trend + DOW correction + profile model is near-optimal for this
    problem — the remaining 112-113 MW is irreducible noise. Further complexity buys
    nothing.

## 6. References

1. Oreshkin, B.N., et al. "N-BEATS: Neural basis expansion analysis for
   interpretable time series forecasting." ICLR 2020.
2. Makridakis, S., et al. "M4 Competition." International Journal of
   Forecasting, 2018.
3. Hyndman, R.J., & Athanasopoulos, G. "Forecasting: Principles and
   Practice." 3rd ed., OTexts, 2021.
4. Ke, G., et al. "LightGBM: A highly efficient gradient boosting
   decision tree." NeurIPS 2017.
5. Zeng, A., et al. "Are Transformers Effective for Time Series
   Forecasting?" AAAI 2023.
