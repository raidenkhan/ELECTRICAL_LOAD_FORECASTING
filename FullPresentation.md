# KNUST × GridCo — Load Forecasting for Ghana's National Grid

---

## Slide 1 | The Data and Why Mature Grid Architectures Fail

**Dataset**

- 70,228 hourly readings from ECG, Jan 2018 – May 2026
- Features available: demand_mw, temperature_c, hour, date

**What the data reveals**

- Demand grew from ~1,467 MW to ~3,750 MW across 8 years (~12.4% CAGR)
- The distribution today is not the distribution from 5 years ago
- A model trained on 2021 data degrades measurably on 2026 data

**Why mature grid models don't transfer**

- European/North American grids have 0–1% annual demand growth with stable distributions
- Their models trained on 5-year-old data still perform well
- West African grids face rapid structural change: electrification, population movement, economic shifts
- Blindly importing architectures from mature grids risks short-lived accuracy

**The hidden factor: electrification**

- Ghana rural electricity access: 67.2% (2018) → 77.8% (2023) → ~83% (2026 est.)
- Hundreds of thousands of new households connect each year with unknown load profiles
- This changes aggregate load shape in ways pure time-series models can't anticipate
- We don't track electrification rate as a model feature — it is absorbed into trend implicitly

---

## Slide 2 | GridCo's Current Approach

- GridCo dispatchers use a semi-manual workflow built around Excel
- They find a similar historical day by experience, adjust for temperature differences, and manually populate the dispatch schedule
- This "similar-day" heuristic is not bad — experienced operators with domain knowledge produce reasonable forecasts
- But it has systematic limitations:

  • Cannot detect patterns that play out over weeks or seasons
  • Does not learn from accumulated historical data
  • No systematic way to improve from past errors
  • Manual process takes ~45 minutes per forecast

**What we validated**

- We digitised GridCo's similar-day approach as a KNN model
- It achieved 141 MW MAE — objectively worse than a simple 2-day weighted trend (113 MW MAE)
- Confirmed that data-driven methods can outperform human heuristics when given enough historical data

**Our goal**

- Provide a more accurate automated forecast to complement GridCo's existing workflow
- Digitise and automate their Excel-based interface so operators spend less time on manual entry and more on high-level decisions
- Keep operators in control — forecast is a suggestion they can accept, edit, or override

---

## Slide 3 | Initial Work — Understanding the Data Before Modelling

- Before building any ML model, we studied what the data itself could tell us

**Patterns we found**

- Strong daily seasonality: morning ramp (6–10h), evening peak (17–21h), overnight trough
- Clear weekly cycle: weekends consistently lower than weekdays
- Monthly variation: dry season vs wet season demand patterns
- Growth trend visible across all years (~12.4% CAGR)
- Temperature correlates with load but changes slowly — its effect is largely captured by hour-of-day and month-of-year features already

**What we chose not to focus on**

- Isolating temperature as a primary driver — it affects load but is slow-moving and collinear with cyclical features
- Overfitting to short-term weather noise — temperature forecasting adds its own uncertainty

**Starting simple**

- We deliberately avoided jumping to deep learning
- First models: classical decomposition (trend + seasonality), weighted trend (yesterday + day-before combined)
- Simple decomposition achieved ~3–4% MAPE as a baseline
- This confirmed the problem was solvable without excessive complexity

---

## Slide 4 | Deep Learning Benchmarking — DLinear Wins

**Evaluation strategy**

- We used a 6-fold expanding-window cross-validation to test how models perform as the grid grows
- Each fold trains on past years and tests on a future year — this simulates real deployment where models must predict unseen future conditions
- The expanding window also reveals which models are robust across years versus models that perform well in one year but fail in others
- This was a key rejection criterion: Transformer and LSTM sometimes looked good on one test year but degraded badly on another

**What we tested**

| Model | Mean MAE across 6 folds | +TIDE correction |
|---|---|---|
| DLinear | 91.0 MW | 67.0 MW (−26.4%) |
| CNN (WaveNet) | 96.8 MW | 74.2 MW (−23.3%) |
| LSTM | 102.3 MW | 78.9 MW (−22.9%) |
| Transformer | 108.7 MW | 82.3 MW (−24.3%) |

- DLinear was the best raw model and stayed best after correction
- The simpler architecture won consistently — deeper models captured noise and spurious patterns
- DLinear has only ~36K parameters (1,000× smaller than a standard vision model), trains on a laptop CPU in minutes

**Why DLinear works (non-technical)**

- It continuously decomposes load into trend (gradual direction) and seasonal (repeating daily/weekly pattern)
- Each component gets a simple learned weight, plus calendar information (hour, day-of-week, month)
- These weighted parts sum to the final forecast — no attention, no recurrence, no convolutions
- The underlying load patterns may not be complex enough to benefit from deeper architectures

---

## Slide 5 | Results — Before Error Correction (Raw DLinear)

**Raw DLinear performance**

| Horizon | MAE | MAPE |
|---|---|---|
| Day-Ahead (24h) | 121 MW | 4.22% |
| Week-Ahead (168h) | 163 MW | 5.3% |
| Month-Ahead (720h) | 104 MW | 3.4% |

- One model handles all three horizons — normally different models would be needed for each
- 720h error can be lower than 168h because longer horizons average out daily variation
- These errors are **before** any correction is applied

**MAE vs MAPE — why both matter**

- **MAE (Mean Absolute Error)**: average error in megawatts — the operational metric GridCo dispatchers care about because they dispatch in MW
- **MAPE (Mean Absolute Percentage Error)**: error as a percentage of actual load — useful for comparing accuracy across different load levels (e.g., 121 MW error is ~3.9% at peak but a larger percentage at low load)
- Both are reported because each tells a different part of the story

**Folds matter — the expanding window reveals model stability**

| Test Year | Raw DLinear MAE |
|---|---|
| 2021 | 166 MW |
| 2022 | 107 MW |
| 2023 | 111 MW |
| 2024 | 226 MW |
| 2025 | 92 MW |
| 2026 | 121 MW |

- The 2024 spike shows how certain years (e.g., COVID recovery) disrupt patterns — a model that worked in 2023 failed in 2024
- This instability is why we tested across 6 folds rather than reporting a single number
- The mean across all folds (110 MW) is more reliable than any one year

---

## Slide 6 | Pushing Further — Learning From Errors

- Even after achieving good raw DLinear results, prediction errors showed systematic patterns
- Errors were correlated across hours (ρ = 0.79 between hour t and t+1)
- This means the residual errors are learnable — we can build a second-stage corrector

**How correction works**

- If DLinear was wrong by +X MW at hour 0, it is likely wrong in the same direction at hour 1
- A corrector predicts this error and subtracts it from the next forecast

**What we tested for correction**

| Corrector | Description | Result |
|---|---|---|
| **TIDE** (Trend-adjusted Iterative Debiasing Engine) | Zero-parameter filter — dampens recent errors with exponentially weighted moving average | **95.5 MW (−20.9%)** on 2026 test |
| **ARDRegression** (batch mode) | Learns error from calendar features (hour, day, month, temperature) — no access to recent error history | No significant improvement |
| **ARDRegression** (sequential mode) | Same model but with access to true recent errors | 68.9 MW (+40% improvement in simulation) |
| SMA-7d / Kalman filter | Traditional smoothing approaches | 98–106 MW (−10% to −19%) |

**Key finding — the availability constraint**

- TIDE's 20.9% improvement comes from one thing: access to recent error history
- ARDRegression with the same features but no error history adds nothing
- This means the dominant error signal is time-based autocorrelation, not weather or calendar patterns
- In the short term, improving bias correctors like TIDE may yield more value than advancing model architecture

**The streaming opportunity**

- The sequential ARD result (+40%) reveals what is possible when a model sees each true error as it happens
- A streaming architecture — correcting forecast hour-by-hour as actuals arrive — could unlock the full error correction potential
- This may be the most practical path forward for grids like ours: a lightweight DLinear model with continuous real-time correction

---

## Slide 7 | Takeaways for West African Grids

**Industrial impact**

- The system is packaged as a web application (FastAPI backend, web dashboard) and handed over to GridCo
- It replaces a ~45-minute manual Excel workflow with a ~1-minute automated process
- GridCo dispatchers use it daily — the forecast auto-fills the dispatch schedule and operators can review, edit, or override
- The model is designed for semi-annual retraining as new data arrives

**What this work says about West African grids**

- Stationary load distributions cannot be assumed — annual growth of ~12.4% means last year's patterns are not this year's
- Models from mature grids (Europe, North America) may perform initially but degrade rapidly in our context
- The 6-fold CV revealed that model rankings change year to year — a model that wins on one test year can fail on another
- Even after achieving ~95 MW MAE with correction, long-term accuracy depends on how the grid evolves

**What we still need**

- Data beyond load and temperature: electrification rates, GDP indicators, urbanisation metrics, industrial activity indices
- Load alone captures the outcome but not the drivers
- Research should go deeper into what factors affect West African grid load beyond historical demand sequences

**Where to go next**

- Short-term: focus on robust, simple correctors (TIDE-style) that work within operational constraints — these give reliable gains without requiring new infrastructure
- Medium-term: incorporate external drivers (electrification, economic data) as model features
- Long-term: a streaming-based architecture where the forecast corrects continuously as actuals arrive may be the best fit for rapidly evolving grids — it exploits the serial correlation that batch models cannot use

**Closing**

- DLinear works because it matches the simplicity of underlying patterns
- TIDE corrects what DLinear misses — bias from rapid growth
- A streaming correction model could unlock the next level of accuracy
- The grid will keep changing, and the research agenda must change with it
