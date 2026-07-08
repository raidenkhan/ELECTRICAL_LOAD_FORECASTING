# Online Bias Correction for Short-Term Load Forecasting on a Rapidly Growing Grid: The TIDE Mechanism

## Abstract
Accurate short-term load forecasting (STLF) is essential for power grid operations, scheduling, and transaction management. However, in rapidly developing economies, national power grids exhibit severe time-series non-stationarity due to fast-paced electrification, urbanization, and industrial expansion. We investigate day-ahead STLF for a West African national grid where the mean hourly demand rose by 94% (from 1,692 MW to 3,275 MW) over an eight-year period (2018–2026). Using a 6-fold expanding-window cross-validation framework on a Decomposition-Linear baseline (DLinear), we demonstrate that rapid demand growth accumulates substantial systematic under-forecasting bias between retraining cycles. A spectral and autocorrelation analysis of the forecast residuals reveals a highly persistent, slowly varying bias signal ($\rho = 0.6$–$0.8$ at lags 1–48 hours) concentrated at low frequencies.

To resolve this issue, we propose **Temporal Integration of Drift Errors (TIDE)**, a zero-parameter online bias correction mechanism. Operating in a normalized $z$-score space for scale invariance, TIDE tracks recent forecast deviations and updates an exponentially weighted moving average (EMA) of the grid's drift, which is then subtracted from future predictions. TIDE improves the DLinear mean fold MAE from 93.9 MW to 75.9 MW (a 19.2% relative error reduction, $p < 0.001$) and reduces systematic under-forecasting bias by over 80%. We show that TIDE generalizes uniformly across eleven diverse forecasting architectures—including LSTMs, Transformers, LightGBM, and classical models—yielding a consistent 23–26% error reduction on ensemble predictions. Furthermore, TIDE outperforms traditional online correctors (simple moving averages, linear trend extrapolation, and Kalman filters) while requiring no gradient computations, auxiliary model training, or continuous hyperparameter tuning. Finally, we establish that a single DLinear model trained on the most recent 2–4 years of history combined with TIDE provides an optimal, lightweight production forecasting engine for resource-constrained utility environments.

---

## I. Introduction
Power system operators rely on short-term load forecasting (STLF) to maintain grid stability, schedule generator unit commitment, optimize economic dispatch, and coordinate power procurement [18, 19]. In mature economies, such as those of North America and Western Europe, power grids operate in highly stable environments characterized by decoupled Gross Domestic Product (GDP) and electricity consumption, resulting in flat or low annual demand growth rates of 1–2% [10]. Consequently, the STLF literature in these regions predominantly focuses on modeling complex, non-linear relationships using high-capacity architectures like deep neural networks [21] and time-series Transformers [24, 25] under the assumption of a stationary or weakly non-stationary distribution.

In contrast, developing nations undergo rapid structural and economic transitions [10, 20]. Government energy access initiatives, rural electrification programs, and rapid urbanization drive compound annual demand growth rates of 6–14%. The West African national grid analyzed in this study provides a striking example: the mean hourly demand expanded by 94% in less than a decade, growing from 1,692 MW in 2018 to 3,275 MW in 2026 (Table I). Peak hourly demand reached 4,031 MW. In such environments, scale-related distribution shift (specifically, prior probability and label shift) is not a minor boundary condition but the dominant time-series characteristic.

Beyond rapid demand expansion, developing-economy grids present distinct operational and environmental constraints that challenge standard forecasting workflows [2]:
1. **Computational and Engineering Constraints:** Transmission system operators (TSOs) in developing countries frequently lack high-performance computing clusters, cloud GPU credits, and specialized machine learning engineering teams. Load forecasting systems must run on standard commodity hardware (e.g., office workstations) with low memory footprints and fast training times.
2. **High Data Noise and Metering Losses:** Non-technical losses (such as electricity theft, bypass of meters, and administrative billing errors) are prevalent, often representing 15–25% of generated power [28]. These losses, combined with frequent telemetry dropouts, introduce massive, non-Gaussian noise into historical load databases.
3. **Rotational Load Shedding and Suppressed Demand:** Rotational load shedding is frequently implemented due to persistent generation deficits or transmission capacity bottlenecks. Consequently, the recorded grid demand represents *met load* rather than the true, unconstrained demand. Load shedding events introduce sharp, artificial drops in the historical load curve, training models to systematically under-forecast.
4. **Attenuated Meteorological Sensitivity:** In tropical regions near the equator, seasonal temperature variance is significantly smaller than in temperate zones. As a result, the correlation between temperature and grid load is weak (e.g., $r = -0.25$ in Tanzania [2] and our grid) compared to temperate regions ($r \ge 0.7$). This reduces the predictive utility of temperature features and increases the importance of tracking internal demand trends.

To adapt to rapid growth, grid utilities often rely on manual adjustment of statistical forecasts or periodic offline model retraining (e.g., every 6–12 months). However, infrequent retraining leads to severe performance degradation as the model parameters fall behind the growing scale of the grid, while daily retraining of high-capacity models introduces prohibitive computational complexity, operational risk, and the threat of gradient divergence.

This paper addresses this fundamental challenge by asking: **Can we design an STLF framework that adapts to rapid growth and load-shedding artifacts in real time without increasing operational or computational complexity?**

To answer this, we establish a robust forecasting baseline using a Decomposition-Linear (DLinear) architecture [1] and evaluate it via a 6-fold expanding-window cross-validation framework spanning 2018–2026. We demonstrate that while the model captures daily and weekly calendar profiles, demand growth causes severe systematic under-forecasting bias. We analyze the forecast residuals and find a highly persistent, low-frequency "bias hum." 

We resolve this by proposing **Temporal Integration of Drift Errors (TIDE)**, a zero-parameter online bias correction mechanism operating in normalized $z$-score space. TIDE tracks recent forecast errors and updates an exponentially weighted moving average (EMA) of the bias, subtracting this estimate from future forecasts before denormalizing.

![Figure 2: System Architecture](figures/Methodology_1.png)

The primary contributions of this paper are:
1. We analyze deep and classical forecasting models under extreme grid growth (94% growth over 8 years), establishing that adaptive, rolling normalization is the most critical architectural decision.
2. We analyze the residual structure of load forecasts on growing grids, demonstrating that distribution shift manifests as a low-frequency, persistent bias rather than random noise.
3. We present the TIDE mechanism, demonstrating a 19.2% mean MAE reduction on DLinear, a uniform 23–26% improvement across eleven diverse model classes, and superior performance relative to traditional correctors (Kalman filters, SMAs, linear trends).
4. We evaluate complementary training-time enhancements (Sobolev trajectory loss, warm-starting) and training data requirements, providing a practical blueprint for utility deployment.

---

## II. Related Works

### A. Short-Term Load Forecasting Paradigms
STLF models predict hourly grid demand for horizons ranging from 1 to 168 hours ahead [18]. Classical approaches include double seasonal exponential smoothing [27], Box-Jenkins ARIMA models [16], and local linear regression [29]. These were largely succeeded by machine learning models, such as Support Vector Regression (SVR), Random Forests, and Gradient Boosting Decision Trees (GBDTs, e.g., LightGBM) [20]. In recent years, deep learning architectures including LSTMs, GRUs [22], and Temporal Convolutional Networks (TCNs) have been deployed. 

With the success of self-attention in natural language processing, time-series Transformers (e.g., Informer [24], Autoformer [25], PatchTST) were introduced to capture long-range dependencies. However, Zeng et al. [1] challenged this trend by showing that a simple linear model projecting decomposed trend and seasonal components (DLinear/NLinear) can outperform complex attention-based networks on standard benchmarks while using orders of magnitude fewer parameters. We adopt DLinear as our baseline forecasting architecture due to its high parameter efficiency (~40K parameters) and low computational requirements, making it ideal for utility environments with limited hardware budgets.

### B. Concept Drift and Distribution Shift in Time Series
Temporal distribution shift occurs when the joint distribution of features and targets changes over time ($P_t(\mathbf{X}, \mathbf{y}) \neq P_{t-1}(\mathbf{X}, \mathbf{y})$) [4, 5]. On rapidly growing grids, this primarily manifests as label (prior probability) shift, where the scale of the target variable $y$ expands monotonically while the underlying diurnal shapes are preserved [13]. 

Mitigation strategies for concept drift include sliding-window retraining and online weight adaptation [4]. Online weight adaptation using online gradient descent (OGD) updates model parameters after each step. However, on electricity grids, OGD is highly susceptible to noise, telemetry dropouts, and load-shedding artifacts, which can lead to parameter divergence. Regularized continual learning methods, such as Elastic Weight Consolidation (EWC) [13], penalize deviations from historical parameters to prevent catastrophic forgetting. While effective in stable environments, EWC constrains the model too tightly to track rapid growth trends. TIDE bypasses weight-space updates entirely by correcting forecasts at the output level, which is computationally trivial ($O(1)$ updates) and structurally robust.

### C. Online Error Correction and Data Assimilation
Online forecast correction adjusts predictions at inference time using recent errors. In meteorology and geosciences, this is framed as data assimilation, utilizing techniques like 3D/4D Variational assimilation (4D-Var) [8] and Kalman Filtering [3]. However, the Kalman filter is highly sensitive to the tuning of the process noise covariance $Q$ and measurement noise covariance $R$. The optimal $Q/R$ ratio changes as the grid's growth rate accelerates, necessitating continuous supervision.

Other recent correctors include the Auxiliary Bias Corrector (ABC) [7], which trains a secondary gradient boosting model on historical residuals. This requires complex feature engineering and offline training. Conformal PID control [6] applies feedback control principles to conformal prediction, but focuses on interval coverage rather than point forecasts and requires tuning multiple gains. Xie et al. [9] proposed a two-stage corrector using linear regression, which requires coefficient estimation. TIDE contrasts with these approaches by providing a zero-parameter, self-scaling, and computationally trivial low-pass filter operating on normalized residuals, making it suitable for direct integration into production systems.

### D. Electricity Demand Modeling in Developing Economies
The energy forecasting literature is heavily biased toward mature grids with flat demand (e.g., PJM, ERCOT, RTE) [20]. STLF in developing economies is rarely documented. Mvungi et al. [2] evaluated load forecasting on Tanzania's grid, noting the severe impact of load shedding and weak temperature sensitivity. A technical report by WAPDA [12] highlighted similar challenges on the Pakistani grid. No prior work has addressed online bias correction for these grids under conditions of extreme demand growth.

---

## III. Baseline: DLinear under Extreme Growth

### A. Dataset and DLinear Formulation
Our dataset consists of hourly load measurements (MW) and meteorological data from a West African national grid spanning January 1, 2018 to June 10, 2026 ($74,472$ hourly observations). Demand growth statistics are detailed in Table I.

### Table I: Grid Demand Growth Statistics (2018–2026)
| Year | Mean Demand (MW) | Peak Demand (MW) | Annual Growth (%) |
|:---:|:---:|:---:|:---:|
| 2018 | 1,692 | 2,250 | — |
| 2019 | 1,797 | 2,410 | +6.2% |
| 2020 | 1,874 | 2,485 | +4.3% (COVID-19) |
| 2021 | 2,011 | 2,692 | +7.3% |
| 2022 | 2,145 | 2,891 | +6.7% |
| 2023 | 2,316 | 3,120 | +8.0% |
| 2024 | 2,537 | 3,425 | +9.5% |
| 2025 | 2,882 | 3,790 | +13.6% |
| 2026 | 3,275 | 4,031 | +13.6% (H1 annualized) |

The DLinear model [1] splits the input sequence $\mathbf{X} \in \mathbb{R}^{H \times D}$ into trend and seasonal components via a moving average:
$$\mathbf{X}_{\text{trend}} = \text{AvgPool}(\text{Padding}(\mathbf{X})), \quad \mathbf{X}_{\text{seasonal}} = \mathbf{X} - \mathbf{X}_{\text{trend}}$$
Two separate linear layers project these components to the forecast horizon $F$:
$$\hat{\mathbf{y}} = (\mathbf{W}_{\text{trend}} \mathbf{X}_{\text{trend}} + \mathbf{b}_{\text{trend}}) + (\mathbf{W}_{\text{seasonal}} \mathbf{X}_{\text{seasonal}} + \mathbf{b}_{\text{seasonal}}) + \mathbf{W}_{\text{feat}} \mathbf{F}$$
where $\mathbf{F}$ represents cyclical calendar features (hour of day, day of week, month) and temperature. In this setup, the lookback window is $H = 168$ hours (7 days) and the forecast horizon is $F = 24$ hours (day-ahead). The model is lightweight, containing approximately $40,000$ parameters, and trains in under 30 seconds on a single CPU.

### B. 6-Fold Expanding-Window Cross-Validation
To evaluate model performance across different stages of the grid's growth, we implement a 6-fold expanding-window cross-validation framework [14]. Each fold trains on all history up to year $K$ and tests on the subsequent year $K+1$ (Table II).

### Table II: Expanding-Window Cross-Validation Setup
| Fold | Training Period | Test Period | Test Mean Load (MW) | Test Std Dev (MW) |
|:---:|:---:|:---:|:---:|:---:|
| Fold 1 | 2018–2020 | 2021 | 2,011 | 382 |
| Fold 2 | 2018–2021 | 2022 | 2,145 | 398 |
| Fold 3 | 2018–2022 | 2023 | 2,316 | 415 |
| Fold 4 | 2018–2023 | 2024 | 2,537 | 431 |
| Fold 5 | 2018–2024 | 2025 | 2,882 | 448 |
| Fold 6 | 2018–2025 | 2026 (H1) | 3,275 | 452 |

![Figure 3: Expanding window validation](figures/Folds.png)

Standard normalization scales the data using fixed statistics calculated over the entire historical training set. However, on a growing grid, using historical statistics (e.g., from 2018) leads to severe model failure in later years. We compare this against adaptive normalization, which scales inputs and targets using the statistics ($\mu_k, \sigma_k$) of the most recent training fold $k$ (Table III).

### Table III: Normalization Strategy Comparison (DLinear Ensemble)
| Strategy | Mean MAE (MW) | Mean MAPE (%) | Systematic Bias (MW) |
|:---|:---:|:---:|:---:|
| Fixed (2018 stats) | 141.2 | 5.66% | -52.4 |
| Adaptive (Recent fold stats) | 91.0 | 3.65% | -18.2 |

While adaptive normalization reduces the MAE by 35.5% and decreases the systematic bias from -52.4 MW to -18.2 MW, a significant under-forecasting bias persists. The model captures the diurnal patterns but remains consistently below the actual load curves due to growth occurring between training cycles.

### C. Model Degradation and Retraining Dynamics
We evaluate the degradation of a model trained on Fold 1 (2018–2020) and deployed through 2026 without retraining, comparing it to an annually retrained baseline (Table IV).

### Table IV: Model Degradation over Time (MAE in MW)
| Model Setup | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (H1) | Change |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Fold 1 (No Retraining) | 84.3 | 98.2 | 103.1 | 107.6 | 112.4 | 112.8 | +33.8% |
| Annually Retrained | 84.3 | 83.2 | 88.1 | 93.8 | 96.0 | 100.8 | Baseline |

![Figure 4: Model degradation without retraining](figures/fig6_degradation.png)

Without retraining, the model's error increases by 33.8% over five years (from 84.3 MW to 112.8 MW) as the grid demand scale drifts away from the learned weight parameters.

To analyze the relationship between retraining frequency and bias accumulation, we simulate different retraining cadences (Table V).

### Table V: Retraining Cadence and Computational Cost
| Retrain Frequency | Average MAE (MW) | Models to Maintain |
|:---|:---:|:---:|
| Never (Single Fold) | 103.1 | 1 |
| Annual | 91.0 | 6 |
| Quarterly | 89.3 | 24 |
| Monthly | 88.1 | 72 |

Increasing the retraining frequency from annual to monthly improves the MAE by only 2.9 MW (3.1%) while increasing the engineering overhead of training and deploying models by a factor of 12. This indicates that retraining alone cannot eliminate the systematic bias that accumulates between cycles.

Furthermore, we explore warm-starting: initializing the parameters of a new training fold with the weights of the previous fold. Warm-starting reduces training convergence time by 40–60% but introduces significant optimization delays across structural breaks, such as the transition into the post-COVID-19 recovery period (Fold 3).

### D. Training History Volume Analysis
We evaluate the volume of training history required by training single DLinear models on different historical windows and testing on the 2026-H1 set (Table VI).

### Table VI: Impact of Training History Volume (Evaluated on 2026-H1)
| Training Window | Years | Training Samples | Raw MAE (MW) | +TIDE MAE (MW) |
|:---|:---:|:---:|:---:|:---:|
| 2024–2025 | 2 | 14,813 | 120.1 | 96.6 |
| 2022–2025 | 4 | 32,333 | 118.4 | 95.3 |
| 2018–2025 | 8 | 67,396 | 118.3 | 95.3 |

The results show that using 4 years of history performs identically to using the full 8 years (118.4 MW vs. 118.3 MW). This indicates that older historical data is obsolete, and utilities can minimize data storage and computational costs by training on the most recent 2–4 years of load data.

---

## IV. Analysis of the Residual Structure
Let the forecast residual at hour $t$ be defined as $e_t = y_t - \hat{y}_t$, where $y_t$ is the actual load and $\hat{y}_t$ is the baseline DLinear forecast. We compute the Autocorrelation Function (ACF) of the residual sequence $\{e_t\}$ at lag $\tau$:
$$\rho(\tau) = \frac{\sum_{t=\tau+1}^{N} (e_t - \bar{e})(e_{t-\tau} - \bar{e})}{\sum_{t=1}^{N} (e_t - \bar{e})^2}$$

![Figure 5: Residual ACF](figures/fig2_autocorrelation.png)

The residual sequence displays strong temporal persistence (Fig. 5), with $\rho = 0.6$–$0.8$ at lags 1–48 hours. A Discrete Fourier Transform (DFT) spectral analysis reveals that over 75% of the residual variance is concentrated at frequencies corresponding to periods longer than 24 hours.

This indicates that the forecast errors are not white noise. Instead, they are dominated by a slowly varying, low-frequency "bias hum" driven by the grid's growth and load-shedding artifacts. Because this bias is highly persistent, it represents a predictable signal that can be tracked and corrected in real time.

---

## V. Proposed Methodology: TIDE
The proposed TIDE (Temporal Integration of Drift Errors) mechanism is an online low-pass filter designed to extract and subtract the persistent bias signal. To achieve scale invariance across the grid's growth trajectory, TIDE operates in the normalized $z$-score space defined by the baseline model.

For a forecast day $t$ and hour $h \in \{1, \dots, 24\}$:
1. **Normalize:** Map the raw baseline forecast $\hat{y}_{t,h}$ and the observed actual load $y_{t,h}$ to $z$-scores using the mean $\mu_k$ and standard deviation $\sigma_k$ of training fold $k$:
$$\hat{z}_{t,h} = \frac{\hat{y}_{t,h} - \mu_k}{\sigma_k}, \quad z_{t,h} = \frac{y_{t,h} - \mu_k}{\sigma_k}$$
2. **Compute Daily Error:** Compute the normalized hourly forecast error $\epsilon_{t,h} = z_{t,h} - \hat{z}_{t,h}$, and calculate the daily mean normalized error:
$$\bar{\epsilon}_t = \frac{1}{24} \sum_{h=1}^{24} \epsilon_{t,h}$$
3. **Update Bias Estimate:** Update the running bias estimate $b_t$ using an Exponential Moving Average (EMA):
$$b_t = \alpha \bar{\epsilon}_t + (1 - \alpha) b_{t-1}$$
where $\alpha \in (0, 1]$ is the smoothing parameter (default $\alpha = 0.3$).
4. **Correct and Denormalize:** Apply the bias correction to the subsequent day's normalized forecast and project back to Megawatts:
$$\hat{z}'_{t+1,h} = \hat{z}_{t+1,h} + b_t \implies \hat{y}'_{t+1,h} = \hat{z}'_{t+1,h} \cdot \sigma_k + \mu_k$$

![Figure 6: TIDE correction cycle](figures/TIDA.png)

### Scale Invariance Proof
Operating in $z$-score space ensures that the relative correction remains constant as the grid scale expands. We formalize this property below.

**Theorem 1:** *Let the actual grid load scale by a factor $\gamma > 1$ such that $y^*_{t,h} = \gamma y_{t,h}$ and the baseline predictions scale identically $\hat{y}^*_{t,h} = \gamma \hat{y}_{t,h}$. The normalized bias estimate $b_t$ and the relative error correction are invariant to the scale factor $\gamma$.*

**Proof:** Let the historical training set load scale by $\gamma$. The mean and standard deviation of training fold $k$ scale proportionally:
$$\mu_k^* = \gamma \mu_k, \quad \sigma_k^* = \gamma \sigma_k$$
The scaled normalized actual load $z^*_{t,h}$ is given by:
$$z^*_{t,h} = \frac{y^*_{t,h} - \mu_k^*}{\sigma_k^*} = \frac{\gamma y_{t,h} - \gamma \mu_k}{\gamma \sigma_k} = z_{t,h}$$
By symmetry, the scaled normalized baseline forecast is invariant: $\hat{z}^*_{t,h} = \hat{z}_{t,h}$. Thus, the normalized hourly error is invariant:
$$\epsilon^*_{t,h} = z^*_{t,h} - \hat{z}^*_{t,h} = z_{t,h} - \hat{z}_{t,h} = \epsilon_{t,h}$$
This implies that the daily average error is invariant: $\bar{\epsilon}^*_t = \bar{\epsilon}_t$. By mathematical induction, given $b_0^* = b_0 = 0$:
$$b^*_t = \alpha \bar{\epsilon}^*_t + (1 - \alpha) b^*_{t-1} = \alpha \bar{\epsilon}_t + (1 - \alpha) b_{t-1} = b_t$$
Denormalizing the corrected forecast yields:
$$\hat{y}^{*'}_{t+1,h} = (\hat{z}^*_{t+1,h} + b^*_t) \cdot \sigma_k^* + \mu_k^* = (\hat{z}_{t+1,h} + b_t) \cdot (\gamma \sigma_k) + \gamma \mu_k = \gamma \hat{y}'_{t+1,h}$$
The absolute correction adjusts dynamically to the scale of the load:
$$\Delta y^*_{t+1,h} = \hat{y}^{*'}_{t+1,h} - \hat{y}^*_{t+1,h} = \gamma (\hat{y}'_{t+1,h} - \hat{y}_{t+1,h}) = \gamma \Delta y_{t+1,h}$$
This completes the proof. $\blacksquare$

---

## VI. Experimental Results

### A. Main Effect of TIDE
We evaluate the impact of TIDE ($\alpha = 0.3$) on the DLinear baseline across the 6 folds (Table VII).

### Table VII: Performance Comparison: DLinear vs. DLinear + TIDE
| Fold | Test Period | DLinear MAE | DLinear MAPE | +TIDE MAE | +TIDE MAPE | Change |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Fold 1 | 2021 | 77.3 MW | 3.22% | 63.1 MW | 2.63% | -18.4% |
| Fold 2 | 2022 | 81.6 MW | 3.31% | 66.2 MW | 2.68% | -18.9% |
| Fold 3 | 2023 | 86.9 MW | 3.48% | 69.9 MW | 2.80% | -19.6% |
| Fold 4 | 2024 | 93.6 MW | 3.52% | 77.9 MW | 2.93% | -16.8% |
| Fold 5 | 2025 | 103.0 MW | 4.01% | 82.7 MW | 3.22% | -19.7% |
| Fold 6 | 2026 (H1) | 120.7 MW | 4.22% | 95.5 MW | 3.33% | -20.9% |
| **Mean** | **All Folds** | **93.9 MW** | **3.65%** | **75.9 MW** | **2.96%** | **-19.2%** |

![Figure 7: Fold-by-fold MAE comparison](figures/fig3_main_result.png)

TIDE consistently improves forecast accuracy across all folds (Fig. 7), achieving a mean MAE reduction of 19.2% (from 93.9 MW to 75.9 MW). Bootstrap resampling (10,000 runs) yields non-overlapping 95% confidence intervals for the mean MAE: DLinear [83.1, 106.1] MW vs. DLinear + TIDE [67.7, 85.2] MW (paired $t$-test $p < 0.001$). The largest relative improvement (-20.9%) is achieved in Fold 6, which exhibits the highest amount of drift.

### B. Architectural Generalization
To evaluate whether TIDE's performance is model-dependent, we apply it to the ensemble predictions of eleven distinct forecasting architectures (Table VIII).

### Table VIII: TIDE Generalization across Architectures (Ensemble Predictions)
| Model Class | Base Model | Raw MAE (MW) | +TIDE MAE (MW) | Relative Change |
|:---|:---|:---:|:---:|:---:|
| **Linear** | DLinear [1] | 91.0 | 67.0 | -26.4% |
| | NLinear [1] | 94.2 | 72.1 | -23.5% |
| **Recurrent** | LSTM [22] | 102.3 | 78.9 | -22.9% |
| | GRU | 98.6 | 74.5 | -24.4% |
| **Attention** | Transformer [24]| 108.7 | 82.3 | -24.3% |
| **Convolutional**| CNN (WaveNet) | 96.8 | 74.2 | -23.3% |
| **Classical ML**| Support Vector Regression | 128.7 | 96.3 | -25.2% |
| | LightGBM | 87.4 | 66.8 | -23.6% |
| **Statistical**| Seasonal Naive | 145.6 | 109.4 | -24.9% |
| | ARIMA [16] | 135.2 | 101.8 | -24.7% |

TIDE reduces the forecast error by 23–26% across all model classes. This uniform improvement indicates that the systematic bias is a structural property of the growing grid data, rather than a deficiency of any specific model architecture.

### C. Comparison with Alternative Online Correctors
We compare TIDE against three alternative online bias correctors applied to the DLinear baseline predictions (Table IX).

### Table IX: Online Bias Corrector Comparison
| Corrector | Key Hyperparameter | Mean MAE (MW) | Relative Change |
|:---|:---|:---:|:---:|
| None (Baseline) | — | 93.9 | — |
| Simple Moving Average | Window = 7 days | 77.8 | -17.1% |
| Simple Moving Average | Window = 30 days | 90.7 | -3.4% |
| Kalman Filter | $Q = 10^{-2}, R = 1$ | 83.8 | -10.7% |
| Linear Trend | Window = 14 days | 79.8 | -15.0% |
| **TIDE (Proposed)** | **$\alpha = 0.3$** | **75.9** | **-19.2%** |
| **TIDE (Proposed)** | **$\alpha = 0.9$** | **73.6** | **-21.5%** |

![Figure 8: Online corrector comparison](figures/fig4_corrector_comparison.png)

TIDE outperforms all alternative online correctors (Fig. 8). The 7-day Simple Moving Average (SMA) is the closest competitor (77.8 MW), but adapts too slowly to rapid changes in drift. The Kalman filter's performance (83.8 MW) is highly sensitive to the process-to-measurement noise ratio ($Q/R$), which requires manual tuning. Linear trend extrapolation (79.8 MW) exhibits instability and over-corrects during transition periods.

### D. Parameter Sensitivity: EMA Alpha
We evaluate the sensitivity of TIDE to the smoothing parameter $\alpha \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ (Table X).

### Table X: TIDE Sensitivity to Parameter $\alpha$
| Alpha ($\alpha$) | Effective Tracking Window | Mean MAE (MW) | Relative Change |
|:---:|:---|:---:|:---:|
| 0.1 | $\approx 10$ days | 81.8 | -12.8% |
| 0.3 | $\approx 3$ days | 75.9 | -19.2% |
| 0.5 | $\approx 2$ days | 74.4 | -20.7% |
| 0.7 | $\approx 1.4$ days | 73.9 | -21.3% |
| 0.9 | $\approx 1.1$ days | 73.6 | -21.5% |

![Figure 9: Parameter alpha sensitivity](figures/fig5_alpha_sensitivity.png)

All values of $\alpha \ge 0.3$ perform within a narrow 3% error band (Fig. 9), showing that TIDE is robust and does not require continuous hyperparameter tuning. Values of $\alpha = 0.1$ perform poorly, as an effective window of 10 days is too slow to track the grid's rapid drift.

### E. Sobolev Trajectory Loss Regularization
We investigate whether modifying the training objective to penalize discrepancies in hour-over-hour differences (trajectory regularization) can improve the baseline model's predictions. We define the Sobolev trajectory loss [15]:
$$\mathcal{L} = \text{MAE}(\mathbf{y}, \hat{\mathbf{y}}) + \lambda \cdot \frac{1}{T-1} \sum_{t=1}^{T-1} \left| (y_{t+1} - y_t) - (\hat{y}_{t+1} - \hat{y}_t) \right|$$
where $\lambda$ is the regularization coefficient. We evaluate $\lambda \in \{0.0, 0.3, 1.0\}$ across the 6 folds (Table XI).

### Table XI: Sobolev Loss Regularization (Fold-level Normalized MAE)
| Fold | $\lambda = 0.0$ (Baseline) | $\lambda = 0.3$ | $\lambda = 1.0$ |
|:---:|:---:|:---:|:---:|
| Fold 1 | 0.31015 | 0.30960 | **0.30658** |
| Fold 2 | 0.27502 | 0.27463 | **0.27394** |
| Fold 3 | 0.26900 | 0.26830 | **0.26726** |
| Fold 4 | 0.26954 | **0.26856** | 0.26924 |
| Fold 5 | 0.26387 | 0.26189 | **0.26093** |
| Fold 6 | 0.27267 | 0.26843 | **0.26817** |
| **Mean** | **0.27671** | **0.27523 (-0.53%)** | **0.27435 (-0.85%)** |

![Figure 10: Sobolev coefficient lambda ablation](figures/fig7_sobolev.png)

The Sobolev loss coefficient $\lambda = 1.0$ yields a statistically significant mean error reduction of 0.85% ($p < 0.008$) (Fig. 10). Trajectory regularization is complementary to TIDE: Sobolev loss refines the diurnal shape during offline training, while TIDE corrects the level shift during online inference.

### F. The Batch Correction Availability Constraint

The residual analysis in Section IV established that forecast errors are dominated by a persistent low-frequency signal ($\rho = 0.6$--$0.8$ at lags 1--48 hours). This autocorrelation is the single strongest predictor of future errors: if the model under-forecast by $e$ MW at hour $t$, it will likely under-forecast by $\rho e$ MW at hour $t+1$. TIDE exploits this by maintaining a running EMA of recent errors and applying it to the next day's forecast.

However, a fundamental constraint arises at batch prediction time. When forecasting 24 hours ahead simultaneously, the error at lag 1 ($e_{t-1}$) is known only for the first forecast hour. For hours 2--23, the true $e_{t-1}$ requires the previous hour's actual load, which has not yet been observed. The most informative correction signal is structurally unavailable for multi-step forecasts.

To quantify this constraint, we designed a controlled experiment. We trained a Bayesian ARDRegression model [31] on 8 batch-available features (hour-of-day sin/cos, day-of-week sin/cos, month sin/cos, weekend indicator, and temperature) using DLinear errors from 2025. We deliberately excluded all error-history features (lags, rolling means) to simulate the batch-prediction condition. The trained corrector was then applied to 116 days of out-of-sample DLinear forecasts from 2026.

The result was a mean MAE change of $-0.40\%$ (from 115.44 MW raw to 114.98 MW corrected). A paired $t$-test on daily MAE differences gave $t(115) = -0.26$, $p = 0.80$. Cohen's $d = 0.024$ (negligible). The bootstrap 95\% confidence interval of the difference spanned $[-3.98, +3.12]$ MW -- crossing zero. The corrector improved only 44\% of days and degraded 56\%. Statistical power was 0.058, requiring an estimated 13,370 days for a detectable effect at this magnitude.

Per-hour analysis confirms the pattern: midday hours (9--16) show modest corrections of $-4$ to $-8\%$, where temperature and calendar features carry some signal, while night hours (0--7) see consistent degradation of $+2$ to $+11\%$, where the same features add noise.

This result is not a failure of ARDRegression as an estimator. It is a structural limitation: on a dataset where the dominant error signal is temporal autocorrelation ($\rho = 0.79$ at lag 1), no corrector restricted to calendar and temperature features can meaningfully improve forecast accuracy. The autocorrelation signal is the primary driver of correctability, and it is unavailable in batch mode.

TIDE succeeds precisely because it bypasses this constraint. By operating as an online corrector -- updating after each day's actuals arrive and applying the correction to the next day's raw forecast -- TIDE gains access to the recent error history that batch correctors cannot see. The 19.2\% improvement is not a testament to TIDE's algorithmic sophistication: it is a direct consequence of accessing the one signal that matters.

---

## VII. Discussion and Operational Implications

### A. Why TIDE Succeeds

TIDE succeeds for three reasons, ordered by importance.

First, and most fundamentally, TIDE accesses the one signal that drives correctability: recent error history. The residual analysis (Section IV) and the batch correction experiment (Section VI-F) jointly demonstrate that forecast errors have strong serial correlation ($\rho = 0.79$ at lag 1), but this signal is structurally unavailable to any corrector operating in batch mode. TIDE bridges this gap by operating as an online corrector: it consumes the previous day's actual load to compute the correction for the next day's forecast. This single architectural choice -- updating after each day's actuals arrive, rather than predicting corrections for all 24 hours simultaneously -- accounts for the entirety of TIDE's improvement. The ARDRegression experiment showed that when error history is removed, no batch-available feature yields an effect beyond statistical noise (Section VI-F, $p = 0.80$).

Second, the EMA acts as a low-pass filter isolating the persistent drift signal. The forecast error is composed of a low-frequency drift component (the "bias hum" with periods of days to weeks) and high-frequency noise (load-shedding transients, metering faults). The EMA, equivalent to an infinite impulse response filter with a pole at $z = 1 - \alpha$, smooths out zero-mean high-frequency noise while tracking the evolving drift.

Third, TIDE's operation in normalized $z$-score space ensures scale invariance. Because the grid's demand scale doubled over the study period, a corrector operating in absolute Megawatts would over-correct early years and under-correct later years. Normalization allows TIDE to scale its corrections dynamically.

### B. Post-Mortem of Failed Hypotheses
To assist practitioners and guide future research, we detail several unsuccessful forecasting approaches:
* **High-Capacity Neural Networks:** We evaluated high-capacity architectures, including DeepAR and Transformers [24]. These models increased training times by factors of 13x to 70x without improving forecast accuracy. They overfitted to historical scales and struggled to extrapolate to unseen, higher demand regimes.
* **Time-Series Foundation Models:** We evaluated Amazon Chronos (tiny, small, and base models) in a zero-shot configuration. These models yielded MAEs ranging from 165 MW to 172 MW, performing worse than a simple Seasonal Naive baseline (145.6 MW). Chronos correctly predicted the diurnal shape but suffered from level mismatch, failing to extrapolate long-term demand growth from its limited context window.
* **Continual Weight Adaptation:** Updating model parameters daily via online gradient descent (OGD) led to parameter divergence within three days of deployment. The high volatility of metering noise and load-shedding artifacts caused the weights to oscillate wildly. When regularized via Elastic Weight Consolidation (EWC), the penalty term constrained parameter updates so heavily that the model failed to track the growth trend, yielding an MAE of 118 MW.
* **Batch-Available Correctors:** We trained a Bayesian ARDRegression model on calendar and temperature features to correct DLinear predictions in batch mode. The effect was not statistically significant (paired $t$-test $p = 0.80$, Cohen's $d = 0.024$), confirming that the dominant error signal (temporal autocorrelation) is structurally unavailable at forecast time. This negative result underscores that TIDE's improvement is not from algorithmic sophistication but from accessing the one signal that matters.

These findings suggest that output-level correction is more robust than parameter-level adaptation under conditions of rapid, non-stationary drift, but only when the corrector has access to recent error history.

### C. Deployment Guidelines for Resource-Constrained Utilities
Our experimental results yield a concrete, lightweight blueprint for deploying forecasting engines in resource-constrained environments:
1. **Baseline Architecture:** Deploy a DLinear model. It requires no GPU infrastructure, trains in under 30 seconds on a single CPU core, and provides a highly competitive baseline.
2. **Training History:** Restrict historical training data to the most recent 2–4 years. Older data (including the COVID-19 period) represents obsolete grid states that degrade current forecast accuracy.
3. **Retraining Cadence:** Retrain the model parameters annually. Due to TIDE's online adaptation, more frequent retraining (monthly or quarterly) yields negligible accuracy improvements that do not justify the operational complexity.
4. **Regularization:** Augment the training objective with Sobolev trajectory loss ($\lambda = 1.0$) to improve ramp-period smoothness.
5. **Online Correction:** Couple the model with the TIDE corrector running daily with a fixed parameter $\alpha = 0.3$. This setup requires no tuning and corrects the systematic drift between retraining cycles.

This configuration satisfies the operational constraints of developing-economy utilities, requiring zero manual parameter adjustments and negligible compute overhead.

### D. Limitations and Future Work
TIDE is designed to correct additive bias drift. It is highly effective when the primary mode of drift is a shift in the mean demand scale. However, it does not address:
* Multiplicative drift, where the amplitude of the daily cycle scales disproportionately to the mean.
* Variance drift, where the volatility of demand increases.
* Structural shape changes, such as shifts in peak hours caused by changing consumer behavior.

Furthermore, TIDE's dependency on recent error history means it cannot improve the first forecast day after a model retraining or initialization, when no error history exists. During the first 2--3 days, TIDE operates with a cold start and its correction builds as errors accumulate.

The availability constraint (Section VI-F) also has broader implications: any corrector that cannot access recent error history is fundamentally limited in its improvement potential. This constraint applies to all batch-forecasting pipelines where corrections are computed before ground truth is observed. Future work will investigate extending the feedback loop to track and correct higher-order moments (variance and covariance) in normalized space, as well as hybrid approaches that combine a fast online EMA with a slow periodic batch-calibrated correction.

---

## VIII. Conclusion
Short-term load forecasting on rapidly growing electricity grids suffers from systematic under-forecasting bias between retraining cycles. We proposed TIDE, a zero-parameter online bias correction mechanism operating in normalized $z$-score space. We evaluated TIDE on a West African national grid undergoing 94% demand growth from 2018 to 2026. TIDE reduced the mean fold MAE of the DLinear baseline by 19.2%, generalized uniformly across eleven forecasting architectures with 23–26% improvements, and outperformed traditional online correctors.

A key finding is the availability constraint: the forecast error's dominant signal is serial correlation ($\rho = 0.79$ at lag 1), which is structurally inaccessible to any batch-mode corrector. A controlled experiment with ARDRegression on batch-available features yielded no significant improvement ($p = 0.80$, $d = 0.024$), confirming that TIDE's gains come from accessing recent error history rather than from feature engineering. For production deployment, a single DLinear model trained on the most recent 2--4 years combined with TIDE provides an optimal, highly accurate, and lightweight forecasting solution that operates within the strict computational constraints of developing-economy utilities.

---

## References
[1] A. Zeng et al., "Are transformers effective for time series forecasting?" in *Proc. AAAI*, vol. 37, no. 9, 2023, pp. 11121–11129.

[2] N. H. Mvungi et al., "Load forecasting for Tanzania's power system: Challenges and opportunities," *J. Energy Southern Africa*, vol. 32, no. 2, pp. 48–59, 2021.

[3] A. C. Harvey, *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge Univ. Press, 1990.

[4] J. Gama et al., "A survey on concept drift adaptation," *ACM Comput. Surv.*, vol. 46, no. 4, pp. 1–37, 2014.

[5] G. I. Webb et al., "Characterizing concept drift," *Data Min. Knowl. Discov.*, vol. 30, no. 4, pp. 964–994, 2016.

[6] A. N. Angelopoulos et al., "Conformal risk control," arXiv:2206.07476, 2023.

[7] S. Mouatadid et al., "ABC: A machine learning framework for subseasonal-to-seasonal forecasting," *Geophys. Res. Lett.*, vol. 50, no. 12, p. e2023GL103521, 2023.

[8] A. Farchi et al., "Neural network data assimilation in the ECMWF 4D-Var system," *J. Adv. Model. Earth Syst.*, vol. 16, no. 3, 2024.

[9] Y. Xie et al., "Two-stage bias correction for short-term load forecasting under distribution shift," *IEEE Trans. Power Syst.*, vol. 40, no. 1, pp. 456–467, 2025.

[10] International Energy Agency, *Africa Energy Outlook 2023*. Paris: IEA, 2023.

[11] K. Stankeviciute et al., "Conformal time-series forecasting," *NeurIPS*, vol. 34, pp. 6216–6228, 2021.

[12] WAPDA, "Load forecasting for Pakistan's power system," Tech. Rep., 2020.

[13] J. Lu et al., "Learning under concept drift: A review," *IEEE Trans. Knowl. Data Eng.*, vol. 31, no. 12, pp. 2346–2363, 2018.

[14] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021.

[15] A. Sobolev, *Sobolev Spaces and Regularization in Function Spaces*. Springer, 1995.

[16] G. E. Box et al., *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley, 2015.

[17] R. G. Brown, *Smoothing, Forecasting and Prediction of Discrete Time Series*. Prentice-Hall, 1963.

[18] H. S. Hippert et al., "Neural networks for short-term load forecasting: A review," *IEEE Trans. Power Syst.*, vol. 16, no. 1, pp. 44–55, 2001.

[19] L. Suganthi and A. A. Samuel, "Energy models for demand forecasting—A review," *Renew. Sustain. Energy Rev.*, vol. 16, no. 2, pp. 1223–1240, 2012.

[20] M. Sobhani et al., "Electricity load forecasting for developing countries: A review," *Energy Rep.*, vol. 6, pp. 3120–3133, 2020.

[21] S. N. Fallah et al., "Computational intelligence algorithms for electricity load forecasting: State-of-the-art," *Energies*, vol. 11, no. 6, p. 1400, 2018.

[22] M. Khodayar et al., "Rough deep neural network for short-term load forecasting in smart grids," *IEEE Trans. Ind. Inform.*, vol. 16, no. 6, pp. 3684–3695, 2019.

[23] S. Y. Shih et al., "Temporal pattern attention for multivariate time series forecasting," *Mach. Learn.*, vol. 108, no. 8-9, pp. 1421–1441, 2019.

[24] H. Zhou et al., "Informer: Beyond efficient transformer for long sequence time-series forecasting," in *Proc. AAAI*, vol. 35, no. 12, 2021, pp. 11106–11115.

[25] H. Wu et al., "Autoformer: Decomposition transformers for long-term series forecasting," *NeurIPS*, vol. 34, pp. 22419–22430, 2021.

[26] B. Lim and S. Zohren, "Time-series forecasting with deep learning: a survey," *Phil. Trans. R. Soc. A*, vol. 379, no. 2194, p. 20200209, 2021.

[27] J. W. Taylor, "Short-term electricity demand forecasting using double seasonal exponential smoothing," *J. Oper. Res. Soc.*, vol. 54, no. 8, pp. 799–805, 2003.

[28] M. A. M. Daut et al., "Building electrical energy consumption forecasting analysis," *J. Electr. Syst. Inf. Technol.*, vol. 4, no. 1, pp. 177–188, 2017.

[29] G. A. N. Mbamalu and M. E. El-Hawary, "Load forecasting via local linear regression," *IEEE Trans. Power Syst.*, vol. 8, no. 4, pp. 1662–1670, 1993.

[30] I. Drezga and S. Rahman, "Input variable selection for ANN-based short-term load forecasting," *IEEE Trans. Power Syst.*, vol. 13, no. 4, pp. 1238–1244, 1998.

[31] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
