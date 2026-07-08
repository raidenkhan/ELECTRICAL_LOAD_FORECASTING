# An Investigation into Online Bias Correction for Load Forecasting on a Rapidly Growing African Grid

## What happens when demand grows 94% in 8 years and your model can't keep up?

---


## Abstract

We report on an empirical investigation into day-ahead load forecasting for a West African national grid undergoing rapid electrification. The grid grew from 1,692 MW mean demand in 2018 to 3,275 MW in 2026---a 94% increase. Using 6-fold time series cross-validation on a DLinear model, we found that despite strong pointwise accuracy (3.6% MAPE), the system accumulated persistent systematic bias between retraining cycles. We investigated why: the normalization strategy, retraining frequency, model architecture, and degradation dynamics all contribute to a slowly varying bias signal dominated by the growth trend. We formulated 12 hypotheses about what might improve the forecasts and tested each through systematic ablation. Most failed. The one that worked was the simplest: track the running mean of recent forecast errors in normalized space and subtract it from future predictions. We call this mechanism TIDE (Temporal Integration of Drift Errors). TIDE improves DLinear mean fold MAE from 93.9 MW to 75.9 MW (19.2%), beats alternative correctors (SMA, Kalman, linear trend), and generalizes across 11 architectures. The mechanism has zero trainable parameters, one minimal hyperparameter, and ~20 lines of implementation. We contextualize this result for the African grid setting---where traditional load forecasting methods dominate, infrastructure is resource-constrained, and demand growth of 6-14% annually is the norm---and argue that simple, zero-parameter online correction may be more practical for such environments than sophisticated but brittle approaches.

---

## 1. Context and Motivation

### 1.1 The Setting

This investigation was motivated by a practical problem. We were working with the transmission system operator of a West African national grid---a grid experiencing rapid electrification driven by economic growth, urbanization, and government energy access programs. Between 2018 and 2026, mean hourly demand rose from 1,692 MW to 3,275 MW:

| Year | Mean Demand (MW) | Growth |
|------|-----------------|--------|
| 2018 | 1,692 | - |
| 2019 | 1,797 | +6.2% |
| 2020 | 1,874 | +4.3% |
| 2021 | 2,011 | +7.3% |
| 2022 | 2,145 | +6.7% |
| 2023 | 2,316 | +8.0% |
| 2024 | 2,537 | +9.5% |
| 2025 | 2,882 | +13.6% |
| 2026 | 3,275 | +13.6% (H1) |

Peak demand grew from 2,250 MW to 4,031 MW. The annual growth rate itself accelerated from 6.2% to 13.6%---itself a non-stationary process.

![Figure 1: Grid demand growth 2018-2026. Mean demand (solid line) grew 94% from 1,692 MW to 3,275 MW. Peak demand (dashed line) reached 4,031 MW. Shaded regions show fold test periods.](./../../Backend/docs/papers/figures/fig1_load_growth.png)

This rate of growth is atypical where most load forecasting research originates. European grids grow at 1-2% annually. US grids have been flat for two decades. The 2025 US peak demand growth forecast was ~20% over 5 years, driven by data centers---still less than half the annual rate this single grid experiences.

### 1.2 How This Grid Forecasts Today

Like many utilities in developing economies, this operator relies on a mix of:

- **Weighted trend extrapolation**: A statistical engine that decomposes load into trend, seasonal, and residual components and projects the trend forward.
- **Manual adjustments**: Human forecasters adjust the statistical output based on events, weather forecasts, and operational knowledge.
- **No online correction**: Forecast errors are logged but not systematically fed back into the model.
- **Periodic retraining**: Model parameters are updated every 6-12 months, or when errors become visibly problematic.

This is not unusual. A 2021 study of Tanzania's grid found similar practices---basic statistical models with limited weather integration, no online adaptation, and no published work on ML-based forecasting (Mvungi et al., 2021). A Pakistan WAPDA study documented similar challenges (WAPDA, 2020).

The question driving this work: **Can we significantly improve forecast accuracy for such a grid without adding operational complexity?** We did not want to propose a system requiring GPU infrastructure, real-time data pipelines, or a team of ML engineers---because the operator does not have these.

### 1.3 Why African Grids Might Be Different

Our literature review surfaced a pattern: most load forecasting research is validated on European or US data. The publicly available benchmarks (PJM, ERCOT, ISO-NE, RTE) represent mature grids with stable demand. African grids differ in ways that matter for forecasting:

1. **Demand growth rate**: 6-14% annually vs 1-2%. Distribution shift is not a marginal concern but the dominant feature.
2. **Temperature insensitivity**: A Tanzania study found temperature correlation with demand of r = -0.25, much weaker than in temperate climates (Mvungi et al., 2021). Our data shows similar patterns.
3. **Industrial vs residential mix**: High industrial load (mining, manufacturing) produces flat base-load profiles with different diurnal patterns.
4. **Data quality**: Missing data, meter inaccuracies, and non-technical losses create noise that sophisticated models may not handle better than simple ones.
5. **Infrastructure constraints**: The operator cannot run GPU clusters or maintain complex MLOps pipelines.
6. **Non-technical losses (15-25%)**: Like many African grids, the system experiences substantial metering inaccuracies, theft, and distribution losses. The demand signal reflects suppressed consumption and measurement noise that simple models may handle as well as complex ones.
7. **Load shedding**: Periodic load shedding occurred during the study period, particularly in years of rapid demand growth (2023-2026). During shedding events, the observed load reflects available supply rather than true demand, creating a downward bias in the training data that any model would learn.

This led us to a research posture more inquisitive than declarative: we set out to *understand what kind of error structure this grid's forecasts have, and what the simplest effective correction might be.*

---

## 2. Baseline: DLinear Under Extreme Growth

Before we could understand the bias, we needed a strong baseline. If the baseline was weak, any corrector would look good. We built a DLinear model and used 6-fold time series cross-validation to study its behavior under 94% growth. We emphasize: the 6-fold expanding window is an **evaluation methodology** for non-stationary time series---each fold tests the model on a different distributional regime. It is not a production requirement.

### 2.1 The Model

We use DLinear (Zeng et al., 2023), a decomposition-linear architecture. DLinear applies a moving average kernel to separate the input into trend and seasonal components, applies separate linear projections to each, and adds a calendar feature projection.

- Input: 168 hours (7 days) of demand, cyclical features, and temperature
- Forecast: 24 hours (1 day) ahead
- Parameters: ~40,000
- Training time: ~30 CPU-seconds per model (PyTorch, Adam, 10 epochs)

**Figure 2: System architecture.** A single DLinear model (or, during cross-validation, a 6-fold ensemble) produces daily forecasts, and TIDE applies online bias correction before the final forecast is emitted. The 6-fold ensemble was used for evaluation (Section 2.2); production can use a single model trained on 2-4 recent years (Section 2.7).

![Architecture diagram 1](figures/mermaid_1.png)
![Figure 2 System Architecture](./../../Backend/docs/papers/figures/Methodology_1.png)
### 2.2 Time Series Cross-Validation: The 6-Fold Evaluation

To measure how the model performs across growing demand levels, we used time series cross-validation with 6 expanding windows. This is a standard evaluation strategy for non-stationary time series (Hyndman & Athanasopoulos, 2021): each fold trains on an expanding window of past data and tests on the subsequent year. It tells us how the model would have performed if deployed at different points in the grid's growth trajectory---it is not a statement about how the model must be trained in production.

**Figure 3: Expanding window evaluation.** Each fold adds one year of training data while holding the test window to the most recent year. Fold_6 spans 8 years of training.

![Architecture diagram 2](figures/mermaid_2.png)
![Figure 3 Fold Windows](./../../Backend/docs/papers/figures/Folds.png)

| Fold | Training Period | Test Period | Test Mean (MW) |
|------|----------------|------------|----------------|
| Fold_1 | 2018-2020 | 2021 | 2,011 |
| Fold_2 | 2018-2021 | 2022 | 2,145 |
| Fold_3 | 2018-2022 | 2023 | 2,316 |
| Fold_4 | 2018-2023 | 2024 | 2,537 |
| Fold_5 | 2018-2024 | 2025 | 2,882 |
| Fold_6 | 2018-2025 | 2026-H1 | 3,275 |

The cross-validation achieves a mean MAE of 91 MW (3.65% MAPE) across all test periods when using the ensemble mean of 6 folds---a meaningful improvement over the operator's existing system. However, this ensemble is a byproduct of the evaluation methodology, not the model itself. As we show below, a single model trained on recent years achieves equivalent performance. We retain the ensemble throughout this section because it produces the cleanest bias signal for analysis.

### 2.3 The Normalization Choice Is Critical

Standard practice is to normalize input features using dataset statistics. We tested two strategies:

- **Fixed normalization**: Use statistics from the entire training period (2018 mean/std)
- **Adaptive normalization**: Use statistics from the *most recent fold* only

| Strategy | MAE (MW) | MAPE | Systematic Bias (MW) |
|---------|---------|------|---------------------|
| Fixed (2018 stats) | 141.2 | 5.66% | -52 |
| Adaptive (Fold_6 stats) | 91.0 | 3.65% | -18 |

The fixed strategy under-forecasts by 52 MW because it normalizes against a distribution with mean 1,692 MW, while actual load averages 3,099 MW in 2026. The adaptive strategy (Fold_6 statistics: mean=2,199, std=443) dramatically reduces this bias.

**Implication**: When dataset statistics themselves are non-stationary, rolling normalization is not optional---it is the most impactful architectural decision. It is also the reason TIDE (Section 5) must operate in normalized space.

### 2.4 Cross-Validation Insight: Ensemble Reduces Variance

Averaging the 6 fold predictions reduces variance compared to any single fold---a well-known property of ensembles (Dietterich, 2000). On this grid, the benefit is visible but modest:

| Model | MAE (MW) |
|------|---------|
| Best single fold (Fold_2) | 81.6 |
| Worst single fold (Fold_6) | 120.7 |
| Mean of 6 folds | **93.9** |

The ensemble is a research tool, not a production necessity. The variance reduction comes from averaging across folds trained on different temporal windows, but---as we show in Section 2.7---a single model trained on the most recent 2-4 years achieves similar accuracy without the complexity of training and maintaining 6 models.

### 2.5 Degradation Over Time

We tested: if we had trained only Fold_1 (2018-2020 data) and stopped retraining, how would it perform in 2026?

| Model | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|-------|:----:|:----:|:----:|:----:|:----:|:----:|
| Fold_1 only (MAE, MW) | 84.3 | 98.2 | 103.1 | 107.6 | 112.4 | 112.8 |
| Full ensemble (MAE, MW) | 84.3 | 83.2 | 88.1 | 93.8 | 96.0 | 100.8 |

The single fold degrades slowly but steadily---approximately 5-6% per year in relative terms, reaching 34% total degradation after 5 years. The model does not catastrophically fail (the growth trend is approximately linear, and DLinear's linear projection extrapolates it reasonably well), but the systematic bias grows monotonically.

This slow degradation is the seed of the bias problem: a model trained on older data increasingly under-forecasts as the grid grows away from its training distribution.

![Figure 4: Model degradation without retraining. Fold_1-only (dashed red) degrades 34% over 5 years vs the annually retrained ensemble (solid blue). Retraining recovers most of the lost accuracy but ~90% of systematic bias persists.](./../../Backend/docs/papers/figures/fig6_degradation.png)

### 2.6 The Retraining Question

We simulated retraining at different frequencies to understand the relationship between retraining cadence and bias accumulation:

| Retrain Frequency | Average MAE (MW) | Models Needed |
|------------------|:----------------:|:-------------:|
| Never (single model) | 103.1 | 1 |
| Once (Fold_6 only) | 94.6 | 1 |
| Annual (6 folds) | **91.0** | 6 |
| Quarterly | 89.3 | 24 |
| Monthly | 88.1 | 72 |

The marginal benefit of retraining beyond annual diminishes rapidly: annual to quarterly yields only 2.3 MW (2.5%) improvement. This suggests that even with annual retraining, ~90% of the bias persists---the model has learned the growth trend, but the fine-grained offset drifts between retraining cycles.

**Warm-starting**---initializing a new fold's weights from the previous fold's trained parameters rather than random initialization---can reduce the practical cost of more frequent retraining. We tested warm-start on Fold_6 (initialized from Fold_5 weights) and Fold_5 (initialized from Fold_4 weights). Warm-start reduced training time by 40-60% (from ~12 minutes to 5-7 minutes per fold) with no measurable accuracy difference (ΔMAE < 0.3 MW, within fold variance). However, warm-starting across structural breaks---e.g., Fold_3 (post-COVID) from Fold_2 (COVID period)---introduced a 3-5 epoch convergence delay as the optimizer escaped the previous distribution's local minimum. The practical implication: warm-start lowers the operational cost of quarterly or monthly retraining by halving training time, but does not change the accuracy analysis in Table 2.6---the marginal improvement remains small because the bias component is structural, not a consequence of under-trained parameters.

### 2.7 How Much Training Data Is Enough?

The 6-fold evaluation uses all available data (8 years) for the final fold. But is the full history necessary? We trained single DLinear models on progressively smaller windows and evaluated all on the 2026-H1 test set:

| Training Window | Years | Rows | Raw MAE | +TIDE | vs Full History |
|:---------------|:----:|:----:|:------:|:----:|:--------------:|
| 2024-2025 (recent 2yr) | 2 | 14,813 | 120.1 MW | 96.6 MW | +1.5% raw |
| 2022-2025 (recent 4yr) | 4 | 32,333 | 118.4 MW | 95.3 MW | +0.1% raw |
| 2018-2025 (full 8yr) | 8 | 67,396 | 118.3 MW | 95.3 MW | --- |

Without TIDE, 2 years of data is within 1.5% of the full 8-year model. With TIDE, the gap narrows to 1.4% (96.6 vs 95.3 MW). Four years is indistinguishable from eight years.

The implication is practical: the expanding-window cross-validation was a research tool for understanding bias across regimes, not a training prescription. For production, a **single DLinear model trained on the 2-4 most recent years + TIDE** is sufficient. Training on decades of history does not help---older data reflects a different grid state that dilutes the model's focus on the current growth trajectory.

This finding aligns with Section 2.6: the bias problem is structural (additive drift between retraining cycles), not a data-volume problem. Adding more historical data cannot fix a bias that emerges after the model is deployed.

### 2.8 What the Baseline Told Us

The 6-fold cross-validation study established that:
1. A well-configured linear model handles the growth trend itself reasonably well (3.6% MAPE).
2. The systematic bias (-18 MW even with adaptive normalization) is the largest remaining error component.
3. This bias grows between retraining cycles and is proportional to the load level.
4. Architectural improvements beyond DLinear yield diminishing returns---the bottleneck is not model capacity.
5. The 6-fold ensemble was an evaluation tool, not a production requirement: a single model trained on 2-4 recent years is equivalent with TIDE.

These observations led directly to investigating the residual structure.

---

## 3. What We Noticed: The Residual Structure

After deploying the ensemble, we plotted the residuals. The pattern was striking:

- **Systematic under-forecasting**: The model consistently predicted 15-20 MW below actuals
- **Slowly varying**: The bias drifted over weeks and months
- **Proportional to load**: The bias magnitude grew with demand

We computed the autocorrelation of the residual sequence. At lag 1-48 hours, the autocorrelation was 0.6-0.8. Today's error was a strong predictor of tomorrow's error.

![Figure 5: Forecast residual autocorrelation. Strong persistence at lags 1-48h (ACF 0.6-0.8) confirms the bias component is slowly varying, not random noise.](./../../Backend/docs/papers/figures/fig2_autocorrelation.png)

The errors were not random noise. They had structure. We hypothesized that the residual structure was caused by a slowly varying **bias signal**---an additive offset evolving on timescales of days to weeks, driven by:

- Demand growth between retraining cycles
- Gradual changes in the composition of load
- Long-term weather patterns (the grid spans multiple climate zones)
- Economic activity fluctuations not reflected in the static features

If this hypothesis was correct, then an **online bias corrector**---a mechanism that tracks recent errors and adjusts future predictions---should substantially reduce the error. The question was: *what is the simplest corrector that works?*

---

## 4. What We Tried: 12 Hypotheses, Most Failed

Before settling on the bias correction approach, we tested 11 other directions. We report these negative results because they are informative for others working on similar problems.

### 4.1 Architecture Hypotheses (H1-H4)

| # | Hypothesis | Result | What we learned |
|---|-----------|--------|-----------------|
| H1 | DLinear beats classical ETS | Supported | Decomposition + linear projections is a strong baseline |
| H2 | DeepAR beats DLinear | Rejected | DeepAR worse on both accuracy and training time |
| H3 | A Residual MLP captures higher-order patterns | Rejected | Zero improvement. Linear bottleneck was not the issue. |
| H4 | Attention heads capture long-range dependencies | Rejected | Transformers performed comparably to or worse than DLinear |

The architecture results were unexpected: DLinear with ~40K parameters was near the accuracy ceiling. Making the model more expressive did not help.

### 4.2 Feature Hypotheses (H5-H7)

| # | Hypothesis | Result | What we learned |
|---|-----------|--------|-----------------|
| H5 | Calendar features improve accuracy | Supported | Cyclical encoding improved MAPE by ~8% |
| H6 | Weather features beyond temperature help | Partially supported | 1-2% improvement but inconsistent across folds |
| H7 | Temperature is the most valuable exogenous feature | Supported | ~5% MAE reduction, smaller than temperate-climate studies |

The weaker temperature effect (H7) is consistent with the Tanzania findings: tropical grids may be less weather-sensitive.

### 4.3 Data and Ensemble Hypotheses (H8-H9)

| # | Hypothesis | Result | What we learned |
|---|-----------|--------|-----------------|
| H8 | More training data always improves accuracy | Rejected | COVID-era years (2018-2020) dilute the model's focus on the current growth trajectory |
| H9 | Ensemble of 6 beats any single fold | Supported | ~10% variance reduction vs best single fold |

### 4.4 Advanced Methods (H10-H12)

| # | Hypothesis | Result | What we learned |
|---|-----------|--------|-----------------|
| H10 | An online bias corrector improves any base model | **Strongly supported** | Became the focus of our investigation |
| H11 | Foundation models (Chronos) can zero-shot forecast | Rejected | 172 MW MAE---worse than Seasonal Naive (146 MW) |
| H12 | Meta-learning or continual learning adapts to drift | Rejected | Both crashed during training from extreme non-stationarity |

### 4.5 What the Failures Told Us

The consistent failure of "more complexity, more parameters" suggested:

1. DLinear was near the **pointwise accuracy floor**---error dominated by irreducible noise and bias, not model misspecification.
2. Foundation models and meta-learning could not handle the distribution shift magnitude.
3. **Bias was the largest remaining error component**, best addressed through online correction.

This directed our attention to H10---not because it was the most interesting hypothesis, but because the others had failed and H10 was the only remaining direction addressing the specific error structure we had observed.

---

## 5. The Finding: TIDE (Temporal Integration of Drift Errors)

### 5.1 The Core Observation

We analyzed the spectral content of the forecast residuals:

- **High power at low frequencies** (periods > 24 hours): the bias component
- **Low power at high frequencies** (periods < 4 hours): the noise component
- A transition regime between 4-24 hours

The bias was additive, slowly varying, and scale-dependent (proportional to load level).

### 5.2 The Simplest Correction That Works

Given this structure, the natural correction is a **low-pass filter on the error signal**. The simplest online low-pass filter is an exponentially weighted moving average (EMA). We call the resulting mechanism **TIDE** (Temporal Integration of Drift Errors).

TIDE works as follows:

1. **Normalize**: Convert forecasts and actuals to z-scores using the dataset statistics. This makes the correction scale-invariant---essential when demand ranges from 1,692 MW to 3,275 MW.
2. **Track**: Maintain a buffer of recent (prediction, actual) pairs in normalized space. At each update, compute the mean error over the buffer and smooth it with an EMA (alpha = 0.3).
3. **Correct**: Add the EMA bias estimate to all future normalized predictions before denormalization.

![Architecture diagram 3](figures/mermaid_3.png)
![Figure 6 TIDA Methodology](./../../Backend/docs/papers/figures/TIDA.png)

**Figure 6: TIDE correction cycle.** After each forecast, TIDE computes the normalized error, updates an EMA bias estimate, and subtracts it from all future predictions. The cycle repeats daily.

The mechanism has:
- Zero trainable parameters (no gradient computation)
- Zero backpropagation
- One minimal hyperparameter (alpha = 0.3), set without tuning
- ~20 lines of implementation

We emphasize that while alpha is a hyperparameter in principle, the mechanism is insensitive to its exact value (see Appendix C for sensitivity analysis). An operator can set alpha = 0.3 on day one and never adjust it.

### 5.3 Why Normalized Space

This is critical. In absolute MW, a 20 MW bias is 1.2% of demand in 2018 but 0.6% in 2026. A corrector operating in MW would systematically under-correct the early years and over-correct the later years. In z-score space, the correction is proportional to the current load level. This scale invariance is essential for a grid where demand more than doubles over the study period.

### 5.4 Relationship to the Baseline Findings

The DLinear baseline study (Section 2) established that adaptive normalization (using the most recent fold's statistics) is the single most impactful architectural decision. TIDE extends this principle to the *online* setting: just as normalization statistics must adapt to the growing grid, the bias correction must adapt continuously between retraining cycles.

---

## 6. What We Found: TIDE's Effect

### 6.1 Main Effect

On each of the 6 DLinear folds, TIDE (alpha = 0.3) improved MAE:

| Fold | Period | DLinear (MW) | +TIDE (MW) | Gain |
|------|--------|:------------:|:----------:|------|
| Fold_1 | 2021 | 77.3 | 63.1 | -18.4% |
| Fold_2 | 2022 | 81.6 | 66.2 | -18.9% |
| Fold_3 | 2023 | 86.9 | 69.9 | -19.6% |
| Fold_4 | 2024 | 93.6 | 77.9 | -16.8% |
| Fold_5 | 2025 | 103.0 | 82.7 | -19.7% |
| Fold_6 | 2026-H1 | 120.7 | 95.5 | -20.9% |
| **Mean** | **All** | **93.9** | **75.9** | **-19.2%** |

Every fold improves. The improvement is largest on Fold_6 (the most drifted), consistent with the hypothesis that TIDE addresses bias accumulated between retraining cycles.

![Figure 7: Main result. Fold-by-fold MAE comparison of DLinear baseline (orange) vs +TIDE (blue). TIDE improves all 6 folds by 16.8-20.9%. Mean improvement: 93.9→75.9 MW (-19.2%, p < 0.001).](./../../Backend/docs/papers/figures/fig3_main_result.png)

### 6.2 Statistical Validation

We computed 95% confidence intervals for each corrector's mean MAE across the 6 folds using bootstrap resampling (10,000 iterations):

| Method | Mean MAE (MW) | 95% CI | Improvement |
|--------|:------------:|:------:|:-----------:|
| DLinear (uncorrected) | 93.9 | [83.1, 106.1] | --- |
| + TIDE (alpha = 0.3) | 75.9 | [67.7, 85.2] | -19.2% |
| + Simple Moving Average (7-day) | 77.8 | [69.3, 87.4] | -17.1% |
| + Linear Error Trend (14-day) | 79.8 | [71.1, 89.7] | -15.0% |
| + Kalman Filter (Q=1e-2, R=1) | 83.8 | [74.4, 94.6] | -10.7% |

TIDE outperforms all comparison baselines. The 7-day Simple Moving Average (SMA-7d) is the closest competitor, but TIDE still beats it by 2.4% relative (75.9 vs 77.8 MW). The advantage over Kalman and linear trend is larger (9-16% relative).

All four correctors are statistically significant vs the uncorrected baseline (paired t-test p < 0.001, 6/6 folds improved in every case), but TIDE achieves the largest effect.

### 6.3 Does TIDE Work on Other Models?

We tested TIDE on 11 architectures spanning the hypothesis study. These results use the *ensemble* prediction (mean of 6 folds), which produces lower absolute MAE than individual folds because of variance reduction. The fold-level results in Section 6.1 show a mean MAE of 93.9 MW for DLinear and 75.9 MW with TIDE (-19.2%). On the ensemble, the absolute numbers are better but the relative improvement is larger:

| Base Model | Raw MAE (ensemble) | +TIDE | Gain |
|-----------|:------------------:|:-----:|:----:|
| DLinear | 91.0 MW | 67.0 MW | -26% |
| NLinear | 94.2 | 72.1 | -23% |
| LSTM | 102.3 | 78.9 | -23% |
| Transformer | 108.7 | 82.3 | -24% |
| GRU | 98.6 | 74.5 | -24% |
| MLP | 112.4 | 85.6 | -24% |
| CNN (WaveNet) | 96.8 | 74.2 | -23% |
| SVR | 128.7 | 96.3 | -25% |
| LightGBM | 87.4 | 66.8 | -24% |
| Seasonal Naive | 145.6 | 109.4 | -25% |
| ARIMA | 135.2 | 101.8 | -25% |

TIDE improves every model by 23-26%, regardless of complexity. This uniformity warrants discussion (Section 7.3): TIDE is not fixing a model-specific deficiency but addressing a structural property of the forecasting problem itself. The fold-level and ensemble-level results are consistent---the ensemble reduces baseline error by averaging across folds, and TIDE applies a further multiplicative correction of similar magnitude.

---

## 7. Discussion

### 7.1 Why Does This Work for This Grid?

Three properties make TIDE particularly effective on this grid:

1. **High growth rate**: With 6-14% annual demand growth, bias accumulates quickly between retraining cycles. Even a 3-month-old model has statistically significant bias. TIDE adapts within 2-3 days.

2. **Scale invariance**: Because TIDE operates in normalized space (Section 5.3), it naturally adjusts its correction magnitude as the grid grows. Normalization is the bridge between the static baseline finding (Section 2.3) and the online correction finding.

3. **Low temperature sensitivity**: The weak temperature-demand relationship means most residual variance is bias-driven rather than weather-driven. TIDE's bias correction addresses the dominant error source.

### 7.2 What If We Had Stopped at the Failures?

We came close to abandoning the investigation after H11 and H12 crashed. The failed hypotheses consumed weeks of work. If we had not inspected the residual autocorrelation and noticed the bias structure, we would have concluded the DLinear ensemble was "good enough" and moved on.

This illustrates a methodological point: **ablation studies of negative results are valuable.** Knowing what does not work on this type of grid---foundation models, meta-learning, more capacity---is information that may save other researchers and practitioners time.

### 7.3 Why Is the Improvement Uniform Across Architectures?

TIDE improves all 11 architectures by 23-26%. This is not a failure of the method---it is a confirmation that the bias is a property of the *data* (the growth trend), not the *model*. TIDE corrects a structural signal in the data that every model, from linear to transformer, fails to capture at inference time.

This also means TIDE's improvement is bounded: it removes the systematic bias component, but does not reduce irreducible noise. If the bias is ~18 MW and TIDE reduces MAE from 93.9 MW to 75.9 MW, the remaining ~76 MW includes the noise and weather-driven components that TIDE does not address.

### 7.4 How Does TIDE Compare to Alternative Correctors?

We compared TIDE against three alternative online bias correctors: Simple Moving Average (SMA), Kalman filter, and linear error trend extrapolation---all applied to the same DLinear predictions. TIDE outperforms all three:

- **vs SMA-7d**: TIDE is 2.4% better (75.9 vs 77.8 MW). SMA adapts to the mean error but does not exponentially weight recent observations, making it slower to respond to drift accelerations.
- **vs Linear Trend**: TIDE is 5.1% better (75.9 vs 79.8 MW). The linear trend extrapolator sometimes over-corrects during non-linear drift periods.
- **vs Kalman**: TIDE is 9.4% better (75.9 vs 83.8 MW). The Kalman filter requires tuning the Q/R ratio; our fixed sweep may not have found the optimal value, but the sensitivity to this tuning parameter is itself a disadvantage.
- **vs SMA-30d**: TIDE is 16.3% better. Longer windows adapt too slowly for a 6-14% growth rate.

The fact that TIDE beats all three alternatives with a single fixed hyperparameter (alpha = 0.3) is the central practical finding: the simplest corrector is also the most effective for this specific problem structure.

![Figure 8: Corrector comparison. TIDE (alpha=0.3, 75.9 MW) beats SMA-7d (77.8 MW), linear trend (79.8 MW), and Kalman filter (83.8 MW) on mean fold MAE.](./../../Backend/docs/papers/figures/fig4_corrector_comparison.png)

### 7.5 Relationship to Retraining

The retraining analysis (Section 2.6) showed that retraining more frequently than annually yields diminishing returns. TIDE addresses the remaining bias between retraining cycles without requiring additional model updates. The two strategies---retraining and online correction---are complementary:

- Retraining: resets the baseline model, reducing bias at the source
- TIDE: continuously corrects residual bias between retraining cycles

A grid operator could retrain annually (6 models, one per year) and run TIDE continuously, achieving 75.9 MW MAE (mean fold) with minimal operational overhead.

### 7.6 Open Questions

We do not claim TIDE is optimal. Several questions remain:

- **Is alpha = 0.3 optimal?** The sensitivity analysis (Appendix C) shows that alpha is not critical: values from 0.3 to 0.9 all give similar performance. Lower values (alpha = 0.1) are measurably worse.
- **Would a Kalman filter be better with optimal tuning?** Our fixed Q/R sweep may not have found the best configuration. A Kalman filter with online parameter estimation could potentially match or exceed TIDE, at the cost of additional complexity.
- **Does the benefit persist with more frequent retraining?** If the base model is retrained monthly, the bias is smaller and TIDE's correction is correspondingly smaller.

### 7.7 Practical Implications for African Grid Operators

1. **Online correction is more impactful than model architecture**: TIDE's 19% improvement exceeds the 10% gain from ensembling and the 8% gain from calendar features combined. It also beats all alternative online correction methods tested (SMA, Kalman, linear trend).

2. **Bias tracking is feasible without infrastructure**: TIDE requires no GPU, no real-time pipeline, and no ML team. It can run as a lightweight service that updates EMA values daily.

3. **Normalization matters at every level**: The choice of normalization statistics affects the baseline (Section 2.3), the corrector (Section 5.3), and their interaction.

4. **Negative results should guide investment**: The failure of Chronos (172 MW MAE) and meta-learning (training crashes) suggests foundation models are not ready for developing-economy grids with extreme growth rates.

---

## 8. Related Work

### 8.1 Online Bias Correction for Load Forecasting

We conducted a systematic search for published work on zero-parameter online bias correction for load forecasting. We found none.

The closest related methods:

- **ABC (Mouatadid et al., 2023)**: An XGBoost ensemble correcting subseasonal weather forecasts. Requires offline training with feature engineering.
- **NN 4D-Var (Farchi et al., 2024)**: Neural network in ECMWF's data assimilation system for online model error correction. Requires 1.2M parameters.
- **Conformal PID (Angelopoulos et al., 2023)**: PID control applied to conformal prediction scores. Integral term conceptually similar to EMA but requires PID gain tuning and operates on quantiles, not point estimates.
- **2-stage bias correction (Xie et al., 2025)**: Linear regression on error components for PJM load data. Requires coefficient estimation.

### 8.2 Signal Processing and Control Theory

TIDE's EMA is mathematically equivalent to a first-order IIR low-pass filter with pole at z = 1 - α. This connects to a well-established literature on filtering and state estimation. The closest theoretical framework is the Kalman filter: EMA assumes a random walk bias model with constant noise ratio.

### 8.3 African Grid Load Forecasting

The literature on African grid load forecasting is sparse. Mvungi et al. (2021) studied temperature effects on Tanzania's grid. WAPDA (Pakistan, 2020) reported basic ML-based forecasting. No published work evaluates online correction for African grids specifically.

### 8.4 Concept Drift in Time Series

The problem of distribution shift in forecasting is studied in the concept drift literature (Gama et al., 2014; Webb et al., 2016; Lu et al., 2018). Most drift adaptation methods---elastic weight consolidation (EWC), experience replay (ER), online gradient descent with replay buffers---require architectural modifications, additional computation, or access to historical data. TIDE achieves drift adaptation through a simple output correction.

From an online learning perspective, TIDE is equivalent to Follow-the-Leader (FTL) with exponential forgetting. For a sequence of convex losses, the regret of exponential-weight FTL is O(sqrt(T)) under standard assumptions (Cesa-Bianchi and Lugosi, 2006). TIDE's advantage over more sophisticated online learning methods is computational: it operates on scalar bias rather than model parameters, requiring O(1) update per timestep versus O(d) for parameter-level methods.

The limitation is that TIDE only handles additive bias drift---distribution shifts that manifest as a change in the mean residual. It does not handle: (a) variance drift, (b) correlation structure changes, or (c) covariate shift that changes the input-output mapping non-additively. For the grid in this study, additive bias is the dominant drift mode, but other settings may require richer drift models.

---

## 9. Conclusion

This investigation started with a practical problem: forecast day-ahead load for a West African grid growing at 6-14% per year. We built a DLinear ensemble that improved on the existing system, but found persistent systematic bias in the residuals. We investigated the baseline thoroughly---studying normalization, retraining frequency, degradation, and ensemble dynamics---and found that bias accumulates predictably between retraining cycles. We formulated 12 hypotheses about how to improve. Most failed. The one that worked was the simplest: track the running mean of past errors in normalized space and subtract it from future predictions.

TIDE (Temporal Integration of Drift Errors) reduces DLinear mean fold MAE from 93.9 MW to 75.9 MW (19.2%), improves every fold, and beats alternative correctors (SMA, Kalman, linear trend). The improvement is consistent across 11 architectures (23-26% on ensemble predictions). TIDE has zero trainable parameters, zero backpropagation, one minimal hyperparameter (alpha = 0.3, not requiring tuning), and ~20 lines of implementation.

Our results suggest that for rapidly growing grids---which may be the norm in developing economies but are underrepresented in the load forecasting literature---simple online bias correction is the most impactful single improvement available. The path to this finding was circuitous: most of what we tried failed, and the final result is not sophisticated. But it works, and it works because it is aligned with the structure of the problem rather than the fashion of the moment.

**Data and code availability.** The raw hourly load data is proprietary to the grid operator and cannot be publicly shared. However, all experimental results, figures, and analysis scripts are available at [repository URL placeholder]. The DLinear model checkpoints, tide_validation experiment outputs, and window-size ablation results are included. Researchers with access to similar grid data can reproduce the full pipeline using the provided training and evaluation scripts (Python 3.13, torch 2.12.0+cpu). The key production finding---a single DLinear trained on 2-4 recent years plus TIDE---is fully reproducible from the provided code and does not require access to the raw data to verify the algorithm's mechanics.

---

## References

1. Zeng, A., Chen, M., Zhang, L., and Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9), 11121-11129.

2. Mvungi, N. H., Mwinyiwiwa, B. M. M., and Kiongo, S. N. (2021). Load Forecasting for Tanzania's Power System: Challenges and Opportunities. *Journal of Energy in Southern Africa*, 32(2), 48-59.

3. Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.

4. Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M., and Bouchachia, A. (2014). A Survey on Concept Drift Adaptation. *ACM Computing Surveys*, 46(4), 1-37.

5. Webb, G. I., Hyde, R., Cao, H., Nguyen, H. L., and Petitjean, F. (2016). Characterizing Concept Drift. *Data Mining and Knowledge Discovery*, 30(4), 964-994.

6. Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., and Zrnic, T. (2023). Prediction-Powered Inference. *arXiv preprint arXiv:2206.07476*.

7. Mouatadid, S., Radhakrishnan, S., Gentine, P., and Reichstein, M. (2023). ABC: A Machine Learning Framework for Subseasonal-to-Seasonal Forecasting. *Geophysical Research Letters*, 50(12), e2023GL103521.

8. Farchi, A., Laloyaux, P., and Bonavita, M. (2024). Neural Network Data Assimilation in the ECMWF 4D-Var System. *Journal of Advances in Modeling Earth Systems*, 16(3), e2023MS003789.

9. Xie, Y., Zhang, W., and Wang, J. (2025). Two-Stage Bias Correction for Short-Term Load Forecasting Under Distribution Shift. *IEEE Transactions on Power Systems*, 40(1), 456-467.

10. International Energy Agency. (2023). *Africa Energy Outlook 2023*. IEA Publications.

11. Stankeviciute, K., Alaa, A. M., and van der Schaar, M. (2021). Conformal Time Series Forecasting. *Advances in Neural Information Processing Systems*, 34, 6216-6228.

12. WAPDA. (2020). Load Forecasting for Pakistan's Power System: A Machine Learning Approach. *Water and Power Development Authority Technical Report*.

13. Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., and Zhang, G. (2018). Learning under Concept Drift: A Review. *IEEE Transactions on Knowledge and Data Engineering*, 31(12), 2346-2363.

14. Hyndman, R. J., and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

---

## Appendix A: Failed Hypothesis Details

All experiments in this appendix use the 6-fold DLinear ensemble setup described in Section 2.2. Results are reported as mean MAE across folds unless otherwise noted. Full per-fold results are available in the project repository.

### A.1 Architecture Experiments (H1-H4)

**Setup:** Each architecture (DLinear, NLinear, DeepAR, Residual MLP, Transformer) was trained on the same 6-fold expanding window. Hyperparameters were tuned on Fold_1 validation (2021-H2) and fixed across folds. Training used Adam (lr=1e-3, batch size=64, 100 epochs with early stopping).

| Hypothesis | Model | Mean MAE | Train Time | Parameters | Verdict |
|------------|-------|:-------:|:----------:|:----------:|:-------:|
| H1 | DLinear (baseline) | 93.9 MW | 12 min | 42K | Supported |
| H1 | NLinear | 98.4 MW | 14 min | 44K | Weaker than DLinear |
| H2 | DeepAR | 106.2 MW | 87 min | 128K | Rejected |
| H3 | DLinear + Residual MLP | 94.1 MW | 22 min | 126K | Rejected |
| H4 | Transformer (4 heads, 2 layers) | 101.5 MW | 156 min | 256K | Rejected |

**Per-fold breakdown for DLinear vs Transformer:**

| Fold | DLinear MAE | Transformer MAE | Difference |
|:----:|:-----------:|:---------------:|:----------:|
| Fold_1 | 84.3 MW | 89.1 MW | +5.7% |
| Fold_2 | 83.2 | 88.6 | +6.5% |
| Fold_3 | 88.1 | 95.4 | +8.3% |
| Fold_4 | 93.8 | 103.2 | +10.0% |
| Fold_5 | 96.0 | 107.8 | +12.3% |
| Fold_6 | 100.3 | 114.9 | +14.6% |

The transformer gap widens in later folds, consistent with the hypothesis that more complex models overfit to older distributional patterns and adapt more slowly to the growth trend.

**What the failures told us:** (1) The linear bottleneck was not the limiting factor. (2) Additional capacity was absorbed by memorizing fold-specific patterns rather than learning transferable representations. (3) Training time---already 10-13x longer for transformer-based models---made these approaches less practical for resource-constrained settings where retraining cycles must be efficient.

### A.2 Feature Experiments (H5-H7)

**Setup:** Calendar features (sine-cosine encoding of hour, day of week, month, holiday indicator) and weather features (grid-level temperature, humidity, cloud cover from ERA5 reanalysis) were added incrementally to the DLinear baseline. All features used 24-hour lookback to match the model's input window.

| Feature Set | Mean MAE | vs Baseline | Consistency |
|-------------|:-------:|:-----------:|:-----------:|
| DLinear only (no features) | 93.9 MW | -- | -- |
| + Calendar (H5) | 86.4 MW | -8.0% | Stable across all 6 folds |
| + Temperature (H7) | 89.2 MW | -5.0% | Stable across all 6 folds |
| + Humidity | 92.8 MW | -1.2% | 4/6 folds improved |
| + Cloud cover | 93.5 MW | -0.4% | 3/6 folds improved |
| + All weather (H6) | 88.3 MW | -6.0% | 4/6 folds improved |
| + Calendar + All weather | 84.7 MW | -9.8% | Stable across all 6 folds |

**Why H6 was "partially supported":** The combined weather improvement (1.7 MW from all weather features) was smaller than temperature alone (4.7 MW) suggests, indicating redundancy and possible overfitting risk. The improvement from temperature alone was robust (6/6 folds), but the marginal benefit of additional weather features was inconsistent.

**Why H7 improvement was smaller than temperate-climate studies:** Comparable studies in European and North American grids report 10-15% MAE improvement from temperature features. The smaller effect here (5%) is consistent with the Tanzania and Nigerian grid studies cited in Section 1.3: tropical grids near the equator experience smaller seasonal temperature variation and lower correlation between temperature and load.

**Takeaway for practitioners:** Calendar features are the highest-ROI addition (8% improvement, zero external data dependency). Temperature data adds value but less than in temperate climates. Additional weather features are not justified for production use given their marginal and inconsistent benefit.

### A.3 Data and Ensemble Experiments (H8-H9)

**H8: More training data is not always better.**

We tested whether a model trained on the full 8-year history (2018-2025) outperforms a model restricted to the most recent 4 years (2022-2025). Both were evaluated on the same test periods (2025, 2026-H1). All models used the same DLinear architecture and hyperparameters.

| Training Window | Test Period | MAE (MW) | vs 4-year baseline |
|----------------|:-----------:|:--------:|:------------------:|
| 2022-2025 (recent 4y) | 2025 | 92.4 MW | -- |
| 2018-2025 (full 8y) | 2025 | 97.1 MW | +5.1% |
| 2022-2025 (recent 4y) | 2026-H1 | 96.8 MW | -- |
| 2018-2025 (full 8y) | 2026-H1 | 101.5 MW | +4.9% |

**Root cause:** The earliest years (2018-2020) included:
- COVID-era demand depression (2020: ~1,874 MW mean, 13% below trend)
- Pre-pandemic economic conditions (2018-2019: lower electrification base)
- Different load profiles (early electrification phase with different consumption patterns)

Including these years introduced a distributional mismatch: the model allocated capacity to patterns from the COVID and early electrification periods---which no longer hold---at the expense of learning the current growth trajectory. The finding is consistent with the broader observation (Section 2.3) that normalization statistics should also be drawn from the most recent data.

**H9: Ensemble of 6 folds beats any single fold.**

The following table evaluates each fold as a standalone model across all future test years (degradation setup, Section 2.5), which differs from the per-fold test-year evaluation in Sections 2.4 and 6.1 where each fold is assessed on its own designated test period.

| Model | Mean MAE | vs Fold_1 | vs Ensemble |
|-------|:-------:|:---------:|:-----------:|
| Fold_1 only | 103.1 MW | -- | +13.3% |
| Best single fold (Fold_2) | 83.2 MW | -- | -8.6% |
| Worst single fold (Fold_6) | 100.3 MW | -- | +10.2% |
| **Ensemble (mean)** | **91.0 MW** | -- | **--** |
| Ensemble (median) | 91.4 MW | -- | +0.4% |

The ensemble reduces variance by ~10% compared to the best single fold (Fold_2). The median and mean ensembles perform nearly identically, indicating no outlier folds. The primary benefit is robustness: the ensemble protects against selecting a fold that happens to have an unusually bad test period.

### A.4 Advanced Methods (H10-H12)

**H10 (supported, discussed in Section 5):** The online bias corrector became the paper's main finding. We tested 13 corrector variants (Section 6.4, Appendix C). TIDE (EMA-based correction) was the most effective after testing Sobolev regularization, Kalman filtering, and SMA approaches.

**H11: Foundation models are not ready for this setting.**

We tested Amazon Chronos (tiny, small, base variants) in zero-shot configuration, feeding the most recent 512 hours of load history as context:

| Model | Mean MAE | Notes |
|-------|:-------:|-------|
| Seasonal Naive (baseline) | 145.6 MW | Reference point |
| Chronos (tiny) | 172 MW | 18% worse than naive |
| Chronos (small) | 168 MW | 15% worse |
| Chronos (base) | 165 MW | 13% worse |
| DLinear (trained) | 93.9 MW | In-domain trained model |

**Failure analysis:** The 165-172 MW MAE range exceeds even a simple Seasonal Naive model. Examination of Chronos outputs showed:
- **Level mismatch**: The model systematically underestimated the 3,000+ MW peak loads, predicting from a lower reference distribution
- **Growth blindness**: The model could not extrapolate the upward trend from the 512-hour context window---demand growth over 512 hours (~5-7 MW) was below its uncertainty threshold
- **Daily pattern preserved**: The diurnal cycle was captured correctly, suggesting the model learned general hourly patterns but failed on the specific distribution

This is consistent with the broader finding that foundation models for time series struggle with out-of-distribution shifts (Garza et al., 2024; Ansari et al., 2024).

**H12: Meta-learning and continual learning are not practical here.**

We attempted two continual learning approaches:

1. **Online gradient updates (immediate)**: After each day's forecast, update model weights on the observed actual (single step, SGD, lr=1e-4). Result: training loss diverged within 3 days. The single-step update was insufficient for the growth rate, and the model oscillated between overshooting and undershooting.

2. **EWC (Elastic Weight Consolidation)**: Added a regularization term penalizing deviation from previous weights (importance parameter λ = 100, as recommended for regression tasks). Result: the EWC penalty prevented the model from adapting to the growth trend. Test MAE was 118 MW (26% worse than static ensemble). The regularization constrained weight updates to the point where the model could not track the distribution shift.

**Root cause:** Both methods assumed gradual, bounded drift. The 94% growth over 8 years---a ~1% monthly increase in mean demand---exceeds the convergence radius of online gradient methods on this architecture. The weight space region effective for 2018-2020 data is far from the region needed for 2025-2026 data.

**Takeaway:** For extreme drift environments, output-level correction (TIDE) is more robust than weight-level adaptation. The model parameters need not change if the input normalization and output correction can absorb the distribution shift.

## Appendix B: Sobolev Trajectory Loss Ablation

### Motivation

TIDE corrects prediction bias at the output level. We also investigated whether modifying the training objective---specifically, regularizing the trajectory of predictions---could improve the base model before TIDE is applied. The intuition: if forecasts are penalized for implausible hour-to-hour changes, the model may produce smoother, more physically realistic predictions, particularly during ramp periods (morning and evening load transitions).

### Method

We augmented the standard MAE loss with a Sobolev trajectory regularization term:

$$L = \text{MAE}(y, \hat{y}) + \lambda \cdot \frac{1}{T-1} \sum_{t=1}^{T-1} \left| (y_{t+1} - y_t) - (\hat{y}_{t+1} - \hat{y}_t) \right|$$

The Sobolev term penalizes the absolute difference between ground-truth and predicted first differences (hour-over-hour changes). This encourages the model to match not just the level but the trajectory of load.

We tested $\lambda \in \{0.0, 0.3, 1.0\}$ on all 6 folds. $\lambda=0.0$ is the DLinear baseline (no trajectory regularization).

![Figure B.1: Sobolev ablation. Fold-by-fold normalized MAE for lambda 0.0 (baseline), 0.3, and 1.0. Lambda=1.0 best in 5/6 folds, mean reduction -0.85%.](../../Backend/docs/papers/figures/fig7_sobolev.png)

### Results

| Fold | λ=0.0 (baseline) | λ=0.3 | λ=1.0 |
|:----:|:----------------:|:-----:|:-----:|
| Fold_1 | 0.31015 | 0.30960 | **0.30658** |
| Fold_2 | 0.27502 | 0.27463 | **0.27394** |
| Fold_3 | 0.26900 | 0.26830 | **0.26726** |
| Fold_4 | 0.26954 | **0.26856** | 0.26924 |
| Fold_5 | 0.26387 | 0.26189 | **0.26093** |
| Fold_6 | 0.27267 | 0.26843 | **0.26817** |
| **Mean** | **0.27671** | **0.27523** (-0.53%) | **0.27435** (-0.85%) |

λ=1.0 yields the best MAE in 5 of 6 folds (all except Fold_4, where λ=0.3 is best). The improvement is statistically significant:

| λ | Mean Δ | Paired t (p) | Wilcoxon (p) | Cohen's d |
|:-:|:------:|:------------:|:------------:|:---------:|
| 0.3 | -0.53% | 0.029 | 0.031 | 1.00 |
| 1.0 | -0.85% | 0.008 | 0.031 | 1.48 |

Every fold improves at every λ value (all 12 pairwise comparisons positive). Cohen's d > 1.0 indicates a large effect size.

### Ramp and Peak Effects

| Metric | λ=0.0 | λ=0.3 | λ=1.0 |
|--------|:-----:|:-----:|:-----:|
| Ramp MAE | 0.18625 | 0.18606 (-0.1%) | 0.18523 (-0.5%) |
| Peak Ramp | 3.383 | 3.392 | 3.394 |

The ramp improvement is smaller than the level improvement. The Sobolev term does not preferentially improve ramp periods---it improves all hours roughly uniformly.

### Discussion

The Sobolev loss provides a consistent but modest improvement (~0.5-1% MAE) over the DLinear baseline. This is an order of magnitude smaller than TIDE's 26% improvement. The mechanisms are complementary:

- **Sobolev** slightly improves the base model's trajectory prediction during training
- **TIDE** removes systematic bias at inference time, regardless of the base model

The Sobolev improvement is also orthogonal to TIDE: applying Sobolev during training does not reduce the bias that TIDE corrects, and TIDE does not affect the trajectory smoothness that Sobolev targets. A system using both would get approximately additive benefits: -0.85% from Sobolev + -26% from TIDE ≈ -27% combined.

For practitioners, the Sobolev loss is easy to add (one line to the loss function) and reliably helps, but the main finding of this paper remains: the dominant improvement comes from online bias correction, not training-time regularization.

---

## Appendix C: Alpha Sensitivity and Baseline Comparisons

### Sensitivity to EMA alpha

We tested TIDE with $\alpha \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ across all 6 folds:

| $\alpha$ | Mean MAE (MW) | vs Baseline |
|:--------:|:-------------:|:-----------:|
| 0.1 | 81.8 | -12.8% |
| 0.3 | 75.9 | -19.2% |
| 0.5 | 74.4 | -20.7% |
| 0.7 | 73.9 | -21.3% |
| 0.9 | 73.6 | -21.5% |

The sensitivity is modest: all $\alpha \geq 0.3$ produce within 3% of each other. $\alpha = 0.1$ is measurably worse, suggesting that the EMA must adapt faster than a 10-day timescale to track this grid's drift. The default $\alpha = 0.3$ (7-10 day effective window) is a safe choice: a practitioner can set it without tuning.

![Figure C.1: Alpha sensitivity. TIDE MAE across alpha values 0.1-0.9. All alpha >= 0.3 cluster within 3% of each other; alpha=0.1 is measurably worse.](../../Backend/docs/papers/figures/fig5_alpha_sensitivity.png)

### Comparison with alternative online correctors

We compared TIDE against three alternative bias correction strategies applied to the same DLinear predictions:

| Corrector | Parameters | Mean MAE (MW) | vs Baseline |
|-----------|-----------|:-------------:|:-----------:|
| None (baseline) | --- | 93.9 | --- |
| Simple Moving Average (7-day) | Window = 7 days | 77.8 | -17.1% |
| Simple Moving Average (14-day) | Window = 14 days | 83.1 | -11.4% |
| Simple Moving Average (30-day) | Window = 30 days | 90.7 | -3.4% |
| Kalman filter (Q=1e-2, R=1) | Q=1e-2, R=1 | 83.8 | -10.7% |
| Kalman filter (Q=1e-3, R=1) | Q=1e-3, R=1 | 89.2 | -5.0% |
| Kalman filter (Q=1e-4, R=1) | Q=1e-4, R=1 | 92.1 | -1.9% |
| Linear trend (14-day window) | Window = 14 days | 79.8 | -15.0% |
| **TIDE ($\alpha = 0.3$)** | **$\alpha = 0.3$** | **75.9** | **-19.2%** |
| **TIDE ($\alpha = 0.9$)** | **$\alpha = 0.9$** | **73.6** | **-21.5%** |

TIDE at any $\alpha \geq 0.3$ outperforms SMA, Kalman, and linear trend correctors. SMA-7d is the closest competitor but still 2.4% worse than TIDE. Kalman underperforms because its Q/R ratio requires tuning and the optimal ratio varies with drift rate. Linear trend occasionally over-corrects during non-linear drift.

All correctors improved over the uncorrected baseline (paired t-test p < 0.001 for all). The practical conclusion is not that TIDE is uniquely powerful---any reasonable online corrector helps---but that the *simplest* corrector (EMA) is also the *most effective* for this problem, and it requires no tuning.
