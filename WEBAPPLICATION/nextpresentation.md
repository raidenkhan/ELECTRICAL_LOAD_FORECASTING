# ECG Load Forecasting — From First Principles to Production

---

## 1. Where We Started: Four Intuitions About Load

We began with a deliberately simple model built on four universal observations about electricity demand:

- **Seasonal profiles** — people use more power at 7 PM than 3 AM. Every hour has a characteristic share of the daily total, and this share varies by month and day-of-week (84 Month×DOW profile shapes).
- **Rolling trend** — more consumers join the grid every year, and economic activity grows. We capture this with a weighted average of the last 1 day (L1) and last 7 days (L7): **Level = 0.65·L1 + 0.35·L7**.
- **Weather influence** — air conditioning drives summer peaks, heating drives winter demand. The Month×DOW profile implicitly encodes the average weather pattern for each month.
- **Momentum** — if demand was high an hour ago, it tends to stay high. Our L1 term gives the most recent 24 hours a 65% weight, effectively acting as an adaptive AR(1) smoother.

**The forecast is simple arithmetic: Forecast = (Level + DOW_offset) × Profile**

The DOW offset adjusts for day-of-week effects: Mon +54 MW, Tue +24 MW, Wed–Fri ~0 MW, Sat –38 MW, Sun –41 MW.

---

## 2. Why This Approach Lasts: Zero Learned Parameters

Unlike a neural network — which has fixed weights learned during training — this engine has **zero learned parameters**. Everything is computed on-the-fly from recent history at prediction time.

- NN: train once, deploy frozen weights → breaks when the data distribution shifts.
- Our engine: **adapts to every new data point** because L1 and L7 are always computed from the most recent demand.

This is inherently durable. Retraining is not needed — you simply upload the latest SCADA CSV and the engine adjusts its level and profiles immediately.

---

## 3. Why Uncertainty Matters: Some Hours Are Harder

Not all hours are equally predictable. Our cross-validated error breakdown by hour shows a clear U-shape:

- **Best hours (03:00–06:00)**: MAE drops to ~88–92 MW. Load is flat and low — the model's level estimate is most accurate here.
- **Worst hours (20:00–24:00)**: MAE climbs to **119–122 MW**. This is the evening ramp-down where disaggregation of commercial, residential, and industrial load creates volatility.

Because of this, we provide **quantile forecasts (80% and 95% confidence bands)** instead of a single point estimate. Day 1: ±125 MW at 80% CI, ±192 MW at 95% CI. Day 7: ±148 MW at 80% CI, ±226 MW at 95% CI.

---

## 4. Why Cross-Validation, Not a Train-Test Split

We evaluate using **6-fold time-series cross-validation**, where each fold is a sequential 6-month block (F1: 2020 H1 through F6: 2025 H1).

**Why CV over a single train-test split?**
- A single split tests only one historical moment. If that period was stable (e.g., 2021 recovery), you overestimate real-world performance.
- CV tests across **six distinct growth regimes** — COVID collapse, stable recovery, moderate growth, high growth, and a **29% CAGR discontinuity** in F6.
- If the model holds up across all regimes, it is genuinely robust — not just memorizing one pattern.

**Result**: MAE ranges from 81 MW (F2, stable recovery) to 123 MW (F6, 29% CAGR), with an aggregate of **98 MW / 4.1% MAPE**. The model never catastrophically fails even in the worst fold.

---

## 5. Deep Learning: Why It Might Not Beat This

We are currently training SOTA deep learning models — **LSTM, GRU, Transformer** — for comparison.

These are still in training, but there are structural reasons they may not beat the baseline — and even if they do, production brings trade-offs.

**Why DL may not outperform:**
1. **Regime shift destroys generalization** — CAGR changes year-to-year: from 4% (stable) to 29% (F6). A model trained on earlier regimes extrapolates poorly into higher-growth periods. Our engine adapts because it recomputes L1/L7 from the most recent 1–7 days.
2. **The data is short and noisy** — only ~7 years of hourly data with sensor gaps, outlier channels, and an outage filter that removed 3.95% of records. DL needs volume.
3. **Profiles are stable, levels shift** — the Month×DOW profile shape barely changes year-to-year; only the level moves. Our engine isolates these two signals. A black-box NN has to rediscover this separation through training.

**Even if DL achieves lower MAE in backtesting, production downsides are significant:**
- **Retraining cost**: every time the grid grows at a new rate, the model needs full retraining.
- **Inference latency**: GPU-dependent, 25–50× slower than a weighted average.
- **Explainability**: a grid operator cannot audit why a transformer predicted a spike. Our model decomposes into Level + DOW + Profile — fully transparent.

**CAGR (Compound Annual Growth Rate)** measures how fast demand grows year-over-year. When CAGR jumps from ~4% to 29% (F6), any model trained on the old rate sees its level estimates systematically lag — a classic "prediction bias" that only an adaptive engine can correct.

---

## 6. Meeting with GRIDCo: What We Learned

We met with GRIDCo and they shared their operational error rates for **24-hour-ahead forecasts**:

| Month | GRIDCo Error | Our Error (D+1) |
|-------|-------------|-----------------|
| January | **1.6%** | ~4.1% |
| February | **1.2%** | ~4.1% |
| March | **~7.0%** | ~4.1% |

**Why GRIDCo might reach such low errors in stable months:** GRIDCo makes a day-ahead forecast and then **refines it intra-day** as actual SCADA readings come in. This is fundamentally different from a one-shot day-ahead prediction — they are effectively doing a rolling 1–6 hour forecast under the hood, then calling it "day-ahead."

We tested this hypothesis with an ablation study comparing **one-shot forecasting** (predict all hours at once, no feedback) vs **daily-update forecasting** (predict 24 hours, then roll forward using latest actuals to recompute the level):

| Strategy | 24h MAE | 168h MAE | 720h MAE |
|----------|---------|----------|----------|
| **One-shot** (no refinement) | **186 MW** | **187 MW** | **190 MW** |
| **Daily update** (refine every 24h) | **114 MW** | **114 MW** | **115 MW** |
| Our production engine (WT+DOW) | **98 MW** | **113 MW** | **148 MW** |

The ablation covers 30 combinations of profile window (7–56 days) × trend window (3–14 days) across 6-fold CV.

**Key finding**: Daily refinement alone cuts MAE by **~40%** (186 → 114 MW). This strongly suggests GRIDCo's low reported errors are at least partly an artifact of intra-day refinement, not an inherently superior model.

But even with daily refinement, the best ablation result (114 MW) does not reach GRIDCo's 1.2% (~44 MW) — meaning they likely have additional advantages: better weather data, shorter actual refinement windows (hourly vs daily), or access to sub-station meter data we lack.

**Critical asymmetry**: GRIDCo achieves 1.2% in stable months but **5× degrades to 7%** in March. Our model does not match their best months but **never exceeds 5% MAPE** — it is consistent across all regimes. For a grid operator, a model that never surprises is often more valuable than one that is sometimes excellent and sometimes unreliable.

We need to investigate GRIDCo's March spike. Possible causes: a sudden demand regime shift, weather anomaly, a change in their internal model, or a data feed issue.

**Important caveat**: 1% of mean demand (~3700 MW) is **37 MW**. Even a "good" 1.2% error means ±44 MW of misallocated capacity. Forecast error must be understood in absolute MW, not just percentage.

**Next step**: Schedule a follow-up meeting with GRIDCo to present our ablation findings, cross-validate our error decomposition against theirs, and discuss how our zero-retrain approach could complement their existing system.

---

## 7. What Is Next

- **Deep learning refinement**: Complete the LSTM/GRU/Transformer training pipeline and compare against baseline. Even if DL is not production-ready, the research benchmark is valuable.
- **GRIDCo alignment**: Share our CV methodology and per-hour error profiles. Understand what caused their March spike.
- **Production hardening**: SCADA upload endpoint is live; next step is automated data freshness monitoring and alerting.
- **Hypothesis**: DL may produce stunning short-term accuracy in backtesting, but for a production system that must survive regime shifts without manual retraining, the adaptive weighted-trend baseline is the pragmatic choice.
