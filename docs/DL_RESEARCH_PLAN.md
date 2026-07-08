# Deep Learning Research Plan — ECG Load Forecasting

> **Status**: Research phase (pre-implementation)
> **Baseline**: Weighted Trend + DOW model — D+1 MAE = 98 MW (4.1% MAPE)
> **Data**: ECG actual demand 2018-01–2026-05-01, hourly resolution, UTC+0
> **Target**: Beat 98 MW D+1 MAE with a testable, reproducible DL hypothesis

---

## Data Characteristics (Known)

| Property | Value | Implication for DL |
|---|---|---|
| Level shift | 2800 → 3850 MW (+38%) over 8 years | Raw DL sees covariate shift |
| Variance growth | σ: 300 → 550 MW (+83%) | Fixed-variance losses mis-calibrated |
| Regime CAGR | –12% (COVID) to +29% (2025 growth) | Single training epoch can't cover all regimes |
| 24h profile shape | Stable across years | Learnable prior exists |
| DOW offset | ±54 MW (Mon +54, Sun –41) | Strong categorical signal |
| Residual autocorrelation | ~0.80 (intra-day smoothness) | NOT exploitable for D+1 — any model faces same limit |
| Leaky floor | 72 MW (shape-only), 82 MW (AR(1)) | Lower bound on any model's D+1 MAE |

---

## Hypothesis 1 (Lowest Risk): WT+DOW + Residual MLP

### Source
Multi-Resolution LSTNet (MDPI Energies 2025); NLinear (NeurIPS 2022); general residual learning

### Concept
Use the existing WT+DOW as a feature extractor (profile, level estimate, DOW offset). Concatenate these features with raw demand and feed into a lightweight 2-layer MLP. The MLP learns to predict the residual error that WT+DOW consistently misses.

### Data Fit
Our residual analysis shows structured error patterns (intra-day shape, level-dependent bias). A small network on top of our working baseline should capture these while the baseline handles the bulk of the forecast.

### Architecture

```
Input (67-dim):
  ├─ raw_24h demand          (24)
  ├─ WT+DOW forecast_24h     (24)
  ├─ DOW one-hot             (7)
  ├─ Month one-hot           (12)
  └─ Profile_24h             (24 at 1am, 24 at 1pm — or full median)

        ↓ concat
   [Linear(67 → 64), ReLU, Dropout(0.1)]
        ↓
   [Linear(64 → 24)]
        ↓
   Residual correction (add to WT+DOW forecast)
```

### Experiment

1. **Training data**: 2018-01–2025-12 residuals (actual − WT+DOW prediction)
2. **6-fold CV**: Same fold split as ablation study
3. **Metrics**: MAE (MW), MAPE (%), fold-to-fold spread
4. **Pass criterion**: Aggregate MAE improvement > 2 MW (i.e., ≤ 96 MW)
5. **Graceful fallback**: If MLP degrades any fold, clip correction at ±2σ or disable per-fold

### Eval Script Template

```python
# Backend/eval_residual_mlp_cv.py
for fold in 1..6:
    train, test = cv_split(fold)
    mlp = ResidualMLPCorrector(hidden_dim=64, dropout=0.1)
    mlp.fit(train_residuals, train_features)
    preds = wt_dow_predict(test) + mlp.predict(test_features)
    mae[fold] = mean_absolute_error(test_actual, preds)
print(f"Baseline MAE: {baseline_mae:.1f}, MLP MAE: {mlp_mae:.1f}, Δ: {delta:.1f} MW")
```

---

## Hypothesis 2 (Low Effort, High Upside): RevIN + LSTM

### Source
RevIN — Kim et al., ICLR 2023 — "Reversible Instance Normalization for Time Series Forecasting"
- Paper: https://openreview.net/forum?id=cGDAkQo1C0p
- Code: https://github.com/ts-kim/RevIN

### Concept
Normalize each input window to N(0,1) before feeding into the network, forecast in normalized space, then de-normalize using the window's original mean and std. Compatible with ANY architecture (LSTM, Transformer, MLP).

### Data Fit
Our level shifts 2800→3850 MW — RevIN strips this away so the network only learns the shape, not absolute level. This mirrors our WT+DOW design (profile × level) but lets the network learn the shape instead of using a precomputed profile. Also handles variance growth (σ 300→550 MW) because normalization removes variance differences too.

### Architecture

```
Input window (168h = 7 days)
        ↓
  [InstanceNorm: (x - mean) / std]
        ↓
  2-layer LSTM (hidden=128, dropout=0.2)
        ↓
  Linear(128 → 24)
        ↓
  [InstanceDenorm: output * std + mean]
        ↓
  24h forecast
```

### Experiment

1. **Compare**: 2-layer LSTM with RevIN vs same LSTM without RevIN
2. **6-fold CV**: Same split
3. **Metrics**: MAE, MAPE, CRPS (probabilistic calibration)
4. **Pass criterion**: Aggregate MAE < 98 MW (beats WT+DOW)
5. **Optional**: Compare learned shape vs precomputed profile

### Eval Script

```python
# Backend/eval_revin_lstm.py
model = RevINWrapper(LSTM(hidden=128, layers=2))
model.fit(train_X, train_y)
preds = model.predict(test_X)
mae = mean_absolute_error(test_actual, preds)
```

---

## Hypothesis 3 (Eval-Only, Zero Investment): Foundation Model Zero-Shot

### Source
- **Chronos-Bolt**: Ansari et al., 2024 — Amazon's time series foundation model
  - https://github.com/amazon-science/chronos-forecasting
- **TimesFM**: Google Research, 2024
  - https://github.com/google-research/timesfm
- **Moirai-2**: Salesforce, 2025
- **Lag-Llama**: Yue et al., NeurIPS 2023
- **ERCOT benchmark**: Simeone, 2026 — Chronos-Bolt achieves MASE 0.31 with 2048h context on Texas load

### Concept
Pre-trained on massive heterogeneous time series corpora, these models claim zero-shot forecasting on unseen domains. If a foundation model already forecasts ECG demand at competitive accuracy, no bespoke training needed.

### Data Fit
The question is: does 2018-2026 Ghana demand distribution appear in the pre-training data? ERCOT (Texas) shows strong performance — Ghana's tropical load pattern (less weather-driven, more economic-growth-driven) is different.

### Experiment

1. **Model**: Chronos-Bolt (smallest variant, fits on consumer GPU/CPU)
2. **Context**: Try 168h, 336h, 720h (7, 14, 30 days)
3. **Test period**: Jan–Apr 2026 (compare to our actual 3.5–5.2% MAPE)
4. **Pass criterion**: MAE ≤ 4.1% (matches WT+DOW) without any fine-tuning
5. **Partial pass**: MAE ≤ 6% (useful as ensemble component)

### Eval Script

```python
# Backend/eval_foundation_zeroshot.py
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",  # or "cuda" if available
)

for test_month in ["2026-01", "2026-02", "2026-03", "2026-04"]:
    context = get_prior_days(test_month, n_days=30)
    forecast = pipeline.predict(context, prediction_length=24 * 30)
    mae = eval(forecast, actuals[test_month])
    report(mae)
```

---

## Hypothesis 4 (Medium Complexity): Online Continual Learning

### Source
- **EWC**: Kirkpatrick et al., PNAS 2017 — Elastic Weight Consolidation
  - https://www.pnas.org/doi/10.1073/pnas.1611835114
- **NatSR**: Urettini et al., 2025 — Natural Score-driven Replay
  - https://www.nature.com/
- **GEM**: Lopez-Paz & Ranzato, NeurIPS 2017 — Gradient Episodic Memory
- **DER++**: Buzzega et al., CVPR 2020 — Dark Experience Replay

### Concept
Maintain a replay buffer of past regime exemplars. Use EWC to penalize changes to weights that were important for previous regimes. This prevents catastrophic forgetting when adapting to a new regime.

### Data Fit
Our CAGR jumps from –12% (COVID) to +29% (2025 growth discontinuity). A model trained on 2018-2024 will catastrophically forget the low-growth regime when fine-tuned on 2025 data. Continual learning explicitly preserves performance across all regimes.

### Architecture

```
[t=0] Train on 2018-2020 (low growth, σ≈350)
    ↓ save importance weights (Fisher information)
[t=1] Fine-tune on 2021-2022 (moderate growth)
    ↓ EWC penalty: λ * Σ F_i * (θ_i - θ*_i)²
[t=2] Fine-tune on 2023-2024 (high growth)
    ↓ EWC penalty + replay buffer sampling
[t=3] Fine-tune on 2025 (extreme growth)
    ↓ EWC penalty + replay buffer sampling
    ↓ Evaluate on ALL regimes
```

### Experiment

1. **Base model**: 2-layer LSTM (same as H2, without RevIN)
2. **Sequential training**: 4 temporal segments (2018-20, 2021-22, 2023-24, 2025)
3. **Compare**: Sequential fine-tuning with vs without EWC
4. **Metrics**: MAE per regime, forgetting measure (MAE on first regime after last training)
5. **Pass criterion**: Fold 6 (worst-case, 29% CAGR) MAE < 110 MW (baseline: 123 MW)
6. **Secondary pass**: Forgetting < 5% MAE degradation on earliest regime

---

## Hypothesis 5 (Highest Risk, Highest Potential): Non-stationary Transformer

### Source
Liu et al., NeurIPS 2022 — "Non-stationary Transformers: Exploring the Stationarization in Time Series Forecasting"
- Paper: https://proceedings.neurips.cc/paper_files/paper/2022/hash/5e2e4c4f4b4f4b4f4b4f4b4f4b4f4b4f-Abstract.html
- Code: https://github.com/thuml/Nonstationary_Transformers

### Concept
Three components:
1. **Series Stationarization**: Whitens input (removes mean and std)
2. **De-stationarized Attention**: Restores variance information via learned MLP that takes `(mean_in, std_in)` and predicts `(mean_out_shift, std_out_scale)`
3. **Stationary Attention**: Preserves dependencies in whitened space

### Data Fit
Our σ grows 300 → 550 MW (+83%) — this is exactly the variance shift that most Transformer architectures ignore. The de-stationarization MLP explicitly models "how does this week's variance affect next week's forecast distribution."

### Architecture

```
Input window (168h)
        ↓
  [Series Stationarization: (x - mean) / std]
        ↓
  [Patch Embedding → Transformer Encoder × N]
        ↓
  [Stationary Attention (softmax(Q*K^T / τ))]
        ↓
  [Series De-stationarization: output * std_hat + mean_hat]
        │                               ↑
        └─── MLP(mean_in, std_in) ───────┘
        ↓
  24h forecast + prediction interval
```

### Experiment

1. **Backbone**: PatchTST (Nie et al., ICLR 2023) or vanilla Transformer
2. **Compare**: With vs without S + D stationarization modules
3. **6-fold CV**: Same split
4. **Metrics**: MAE, CRPS (continuous ranked probability score), interval width
5. **Pass criterion**: Improved CRPS over WT+DOW (better probabilistic calibration when σ is high)
6. **Secondary**: Aggregate MAE ≤ 95 MW

---

## Hypothesis 6 (Research Deep-Dive): Meta-Learning Fast Adaptation

### Source
- **DeepTime**: Woo et al., ICLR 2023 — "DeepTime: Time Series Forecasting as a Continuous-Time Optimization"
  - https://openreview.net/forum?id=7b7YEhP2zP
- **DTAM**: Darzi et al., ICLR 2025 Workshop — "Deep Time Adaptation Models"
- **STaRNet**: Applied Soft Computing, 2025
- **MAML**: Finn et al., ICML 2017 — Model-Agnostic Meta-Learning
- **Reptile**: Nichol et al., 2018 — "On First-Order Meta-Learning Algorithms"

### Concept
Train across many "tasks" (e.g., each year is a task) via MAML or Reptile. The model learns an initialization that can adapt to a new regime in just a few gradient steps. DeepTime uses a learned function of continuous time rather than discrete windows.

### Data Fit
Our profiles are stable but levels shift. Meta-learning could learn the "profile prior" across years and adapt the "level parameter" in a few steps — directly analogous to our fixed profile × adaptive level formula, but parameterized as a neural network.

### Architecture

```
Meta-training (loop over tasks T₁, T₂, ..., Tₙ):
  1. Sample task Tᵢ (e.g., year 2018 data)
  2. Inner loop: θ' = θ - α ∇_θ L_Tᵢ(f_θ(support_set))
  3. Meta-objective: min_θ Σ L_Tᵢ(f_θ'(query_set))
  4. Update: θ ← θ - β ∇_θ Σ L_Tᵢ(f_θ')
     (or Reptile: θ ← θ + ε (θ' - θ))

Meta-testing (new regime):
  θ_init = meta-learned initialization
  θ_adapt = θ_init - α ∇ L(f_θ_init(first_7_days))
  forecast next 24h with f_θ_adapt
```

### Experiment

1. **Model**: Time-index MLP (DeepTime-style) or simple LSTM
2. **Task definition**: Each year (2018–2025) = one task
3. **Inner steps**: 5–10 gradient steps
4. **Evaluation**: Given first 7 days of a new fold, how fast does it converge to 4.1% MAPE?
5. **Pass criterion**: Convergence to ≤ 4.1% MAPE within ≤ 7 days of new regime data
6. **Comparison**: Meta-learned initialization vs random initialization vs WT+DOW

---

## Hypothesis 7 (DLinear-Specific): Per-Hour Weighted Loss

### Source
DLinear ablation findings (this repo); MAE distribution shows 85 MW (hour 2) → 167 MW (hour 24)

### Concept
DLinear's L1 loss treats all 24 forecast hours equally. But the late-hour error (167 MW) is nearly 2× the early-hour error (85 MW). A weighted loss that penalizes late hours more during training should force the model to allocate capacity to the harder-to-predict hours.

### Data Fit
The per-hour MAE increase is monotonic: hour 1-6 ≈ 90 MW, hour 7-12 ≈ 120 MW, hour 13-18 ≈ 145 MW, hour 19-24 ≈ 160 MW. The evening ramp-up (17-20h) is the most volatile period. A weighted loss with linearly increasing weights (1.0 at hour 1 → 2.0 at hour 24) would rebalance the optimization.

### Experiment

1. **Base model**: Identical DLinear (40K params, k=25, 168h window)
2. **Loss variants**:
   - L1 uniform (baseline) — 91 MW
   - L1 linear weight w_h = 1.0 + (h-1) * (1.0/23) — hour 1: 1.0, hour 24: 2.0
   - L1 quadratic weight w_h = 1.0 + ((h-1)/23)² — penalizes late hours more aggressively
   - Weighted + late-hour oversampling — sample late-hour errors more frequently
3. **Metrics**: Per-hour MAE profile (target: flatten the curve), aggregate MAE
4. **Pass criterion**: Aggregate MAE < 91 MW AND late-hour MAE (hours 19-24) < 140 MW
5. **6-fold CV**: Same splits as DLinear benchmark

---

## Hypothesis 8 (DLinear-Specific): Multi-Kernel Decomposition

### Source
DLinear (Zeng et al., AAAI 2023); FEDformer (Zhou et al., ICML 2022); SCINet (Liu et al., NeurIPS 2022)

### Concept
DLinear decomposes demand into trend + seasonal using a single moving average (k=25). But demand has multiple periodicities: daily (k=24), weekly (k=168), and monthly (k=720). Using multiple kernel sizes captures each scale separately, allowing the linear projections to specialize.

### Data Fit
ECG demand shows three clear periodicities: daily profile (24h peak-trough), weekly pattern (Mon-Fri higher, Sat-Sun lower), and annual/seasonal shift (dry season vs rainy season). Each operates at different timescales — a single kernel can't separate them.

### Architecture

```
Input demand (B, 168)
     ↓
  [MA kernel 24] → trend_24 → Linear → 24h
  [MA kernel 168] → trend_168 → Linear → 24h
  [MA kernel 720] → trend_720 → Linear → 24h (padded for short window)
  [Residual] → seasonal → Linear → 24h
     ↓
  Sum all 4 outputs → 24h forecast
```

### Experiment

1. **Base model**: DLinear (k=25) — 91 MW
2. **Multi-kernel variant**: kernels = [24, 168, 720], each with its own Linear head
3. **Compare**: DLinear vs MultiKernel-DLinear per-fold and per-hour
4. **Metrics**: MAE, per-hour profile, parameter count
5. **Pass criterion**: Aggregate MAE < 88 MW (3 MW improvement over baseline DLinear)

---

## Hypothesis 9 (DLinear-Specific): Quantile DLinear

### Source
Quantile Regression (Koenker & Hallock, 2001); SQR (Xu et al., 2025); QD-DLinear (this repo)

### Concept
Replace the point-forecast output head with a quantile output head (10th, 50th, 90th percentiles). This adds uncertainty quantification to DLinear — matching WT+DOW's ±125 MW / ±192 MW uncertainty bands. The 50th percentile (median) should match or exceed the current point forecast accuracy.

### Data Fit
WT+DOW provides 80% confidence bands (±125 MW D+1, ±192 MW D+7). Grid operators need uncertainty information for reserve planning. DLinear currently gives only a point forecast — adding quantiles makes it a direct replacement.

### Architecture

```
DLinear backbone (unchanged)
     ↓
  3 output heads instead of 1:
    ─ Linear_10: outputs 10th percentile (B, 24)
    ─ Linear_50: outputs 50th percentile (B, 24)  
    ─ Linear_90: outputs 90th percentile (B, 24)
     ↓
  Quantile loss: Σ ρ_τ(y - ŷ_τ) where ρ_τ(u) = u * (τ - I(u < 0))
```

### Experiment

1. **Base model**: DLinear (point forecast) — 91 MW
2. **Quantile variant**: DLinear with 3 output heads (τ = 0.1, 0.5, 0.9)
3. **Compare**: Median (τ=0.5) MAE vs point forecast MAE; 80% interval coverage
4. **Metrics**: MAE (median), interval coverage (nominal 80%), interval width
5. **Pass criterion**: Median MAE ≤ 93 MW AND 80% interval achieves ≥ 75% empirical coverage
6. **Caveat**: Quantile DLinear may have slightly worse point accuracy — the trade-off is uncertainty info

---

## Hypothesis 10 (DLinear-Specific): Adaptive Level Correction

### Source
WT+DOW dual-weight level formula (this repo); online adaptation in N-BEATS (ElementAI, 2020)

### Concept
DLinear is trained on fixed data windows. As the level shifts (2800 → 3850 MW over 8 years), the static model degrades. Add a lightweight adaptive correction: compute the mean error over the last 24-48h and apply it as a bias correction to the DLinear output. This mirrors WT+DOW's L1/L7 level tracking.

### Data Fit
DLinear degrades ~7 MW/year as training data becomes stale (Fold 1: 80 MW at 2020 test vs Fold 5: 107 MW at 2024 test). An adaptive level correction should recover most of this degradation without retraining.

### Architecture

```
DLinear forward pass (unchanged):
  x (168, 9) → DLinear → raw_forecast (24,)

Adaptive correction (online, no training):
  Compute: recent_bias = mean(actual[-48:] - raw_forecast[-48:])
  Apply:   final_forecast = raw_forecast + recent_bias
           (exponential moving average: bias = α * bias + (1-α) * new_bias)
```

### Experiment

1. **Base model**: DLinear (static, no correction) — 91 MW
2. **Variants**:
   - DLinear + 24h bias correction (α=0.3)
   - DLinear + 48h bias correction (α=0.5)
   - DLinear + EMA bias (α=0.15)
3. **Test**: Simulate production daily forecasts with actual feedback
4. **Metrics**: MAE per fold, degradation over time since last retrain
5. **Pass criterion**: DLinear + correction beats static DLinear on ALL 6 folds
6. **Key advantage**: Extends retrain interval — model stays accurate longer

---

## Hypothesis 11 (DLinear-Specific): Learnable Hour Embeddings

### Source
Time2Vec (Kazemi et al., 2019); positional encoding variants (Vaswani et al., 2017)

### Concept
DLinear currently uses fixed sin/cos embeddings for hour, DOW, and month features. Learnable embeddings can capture more complex temporal patterns — e.g., the 17:00-19:00 peak shape differs fundamentally from the 03:00-05:00 trough shape, but sin/cos encodes them with the same functional form.

### Data Fit
The per-hour MAE pattern (85 MW at hour 2 → 167 MW at hour 24) suggests that DLinear's sin/cos features don't adequately represent the asymmetric daily profile. Learnable hour embeddings can specialize each hour's representation.

### Architecture

```
Replace sin/cos features with:
  hour_embed:   Embedding(24, 8)  → 8-dim learned vector per hour
  dow_embed:    Embedding(7, 4)   → 4-dim learned vector per DOW
  month_embed:  Embedding(12, 4)  → 4-dim learned vector per month

Input becomes: demand + hour_embed + dow_embed + month_embed + temp + holiday
              = 1 + 8 + 4 + 4 + 1 + 1 = 19 features (up from 9)

All other DLinear architecture unchanged.
```

### Experiment

1. **Base model**: DLinear (sin/cos features, 9 channels) — 91 MW
2. **Embed variant**: DLinear (learnable embeddings, 19 channels)
3. **Compare**: Per-fold MAE, per-hour profile, training stability
4. **Metrics**: MAE, parameter count, convergence speed
5. **Pass criterion**: Aggregate MAE < 89 MW (2 MW improvement)

---

## Hypothesis 12 (DLinear-Specific): Distilled DLinear for Edge Deployment

### Source
Hinton et al., 2015 — "Distilling the Knowledge in a Neural Network"

### Concept
DLinear has 40K parameters. Can we distill it to < 10K params while maintaining ≥ 95% of accuracy? A smaller model means faster inference, lower power, and potential deployment on embedded hardware at substations.

### Data Fit
DLinear is already small (40K params). But for edge deployment at 50+ substations, every kilobyte matters. The signal in load forecasting is fundamentally simple — 40K params may be over-parameterized for this task.

### Architecture

```
Teacher: DLinear (40K params, 91 MW)
    ↓ (generate soft targets on training data)
Student: TinyDLinear (4-10K params)
  ─ Single linear layer, no decomposition
  ─ Or: DLinear with hidden_dim=16 instead of 40
    ↓
Loss = α * L1(student, actual) + (1-α) * KL(student_logits, teacher_logits)
```

### Experiment

1. **Teacher**: DLinear (40K params, 91 MW)
2. **Students**: 
   - TinyDLinear-A: single Linear(168*9 → 24) = 36K params (no decomposition)
   - TinyDLinear-B: DLinear with 8-dim hidden = 8K params
   - TinyDLinear-C: DLinear with 4-dim hidden = 4K params
3. **Distillation**: Train students on teacher soft targets + actual L1
4. **Metrics**: MAE, parameter count, inference time (CPU)
5. **Pass criterion**: Student < 10K params with MAE < 94 MW (≥ 95% of teacher performance)

---

## Updated Execution Order

| Step | Hypothesis | Risk | Effort | Expected Gain | Pass Threshold | Depends On |
|------|-----------|------|--------|---------------|----------------|------------|
| 1 | H1: WT+DOW + residual MLP | Lowest | 1-2 days | 2-5% MAE cut | > 2 MW reduction | WT+DOW baseline |
| 2 | H2: RevIN + LSTM | Low | 2-3 days | 10-20% MAE cut | MAE < 98 MW | — |
| 3 | H7: DLinear weighted loss | Low | 1-2 days | 2-5% on late hours | Late-hour MAE < 140 MW | DLinear checkpoints |
| 4 | H10: DLinear adaptive correction | Low | 1 day | 3-5% MAE cut | Beat static on all folds | DLinear checkpoints |
| 5 | H11: DLinear hour embeddings | Low | 2-3 days | 2-5% MAE cut | MAE < 89 MW | DLinear baseline |
| 6 | H8: DLinear multi-kernel | Medium | 2-3 days | 3-5% MAE cut | MAE < 88 MW | DLinear baseline |
| 7 | H9: DLinear quantile outputs | Medium | 2-3 days | 0% point MAE (uncertainty) | 75% coverage at 80% nominal | DLinear baseline |
| 8 | H3: Foundation model zero-shot | Low (eval) | 1 day | 0% (baseline comparison) | MAE ≤ 4.1% | Foundation model download |
| 9 | H12: Distilled DLinear | Low | 1-2 days | 0% (size reduction) | < 10K params at 94 MW | DLinear checkpoints |
| 10 | H4: Online continual learning | Medium | 3-5 days | 15-25% on extreme folds | Fold 6 MAE < 110 MW | H1 or H2 architecture |
| 11 | H6: Meta-learning | High | 5-7 days | 10-20% | ≤ 7 days to converge | — |
| 12 | H5: Non-stationary Transformer | High | 5-7 days | 10-15% + better calibration | CRPS improvement | — |

> **Note**: H7-H12 are DLinear-specific and can run in parallel with H1-H3.
> They reuse existing DLinear checkpoints and require no GPU retraining
> (except H8, H11 which need training from scratch).

---

## Implementation File Structure (Proposed)

```
Backend/
├── app/
│   └── ml/
│       ├── weighted_trend_engine.py      # existing
│       ├── residual_mlp_corrector.py     # H1 — new
│       ├── revin_lstm.py                 # H2 — new
│       ├── foundation_zeroshot_eval.py   # H3 — new (eval only)
│       ├── online_continual_learner.py   # H4 — new
│       ├── meta_learning.py              # H6 — new
│       └── nonstationary_transformer.py  # H5 — new
├── eval_residual_mlp_cv.py               # H1 eval — new
├── eval_revin_lstm.py                    # H2 eval — new
├── eval_foundation_zeroshot.py           # H3 eval — new
├── eval_continual_learning.py            # H4 eval — new
├── eval_meta_learning.py                 # H6 eval — new
└── eval_nonstationary_transformer.py     # H5 eval — new
```

---

## Reference Papers

| Paper | Venue | Key Idea |
|-------|-------|----------|
| RevIN (Kim et al.) | ICLR 2023 | Instance normalization for TS |
| Non-stationary Transformer (Liu et al.) | NeurIPS 2022 | Stationarization + de-stationarization |
| Chronos (Ansari et al.) | 2024 | Foundation model for TS |
| EWC (Kirkpatrick et al.) | PNAS 2017 | Elastic weight consolidation |
| DeepTime (Woo et al.) | ICLR 2023 | Continuous-time meta-learning |
| NLinear (Zeng et al.) | NeurIPS 2022 | Simple linear beats Transformer |
| PatchTST (Nie et al.) | ICLR 2023 | Patching for time series |
| MAML (Finn et al.) | ICML 2017 | Model-agnostic meta-learning |
| DER++ (Buzzega et al.) | CVPR 2020 | Dark experience replay |
| GEM (Lopez-Paz & Ranzato) | NeurIPS 2017 | Gradient episodic memory |

---

## Success Criteria

1. **Primary**: Any DL hypothesis beats WT+DOW aggregate D+1 MAE (98 MW / 4.1%) on 6-fold CV
2. **Secondary**: The improvement is consistent (≥ 4 of 6 folds show improvement)
3. **Tertiary**: The model does not catastrophically fail on any fold (no fold MAE > 150 MW)
4. **Operational constraint**: Model can run inference on the same CPU hardware as WT+DOW (< 100ms per forecast)
5. **Reproducibility**: All experiments use fixed seeds, documented splits, and produce eval scripts that can be re-run

---

## Appendix: Reproducibility Checklist

- [ ] Fixed random seed (42) for all experiments
- [ ] Same 6-fold CV split as ablation study
- [ ] Training/validation/test separation logged per run
- [ ] Hyperparameters recorded in experiment log
- [ ] GPU vs CPU inference time reported
- [ ] Model checkpoint saved with eval script
- [ ] Per-month metrics reported (not just aggregate)
- [ ] Fold-by-fold breakdown reported
