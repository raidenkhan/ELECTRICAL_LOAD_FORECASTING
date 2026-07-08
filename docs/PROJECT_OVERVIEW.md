# GRIDCo Achimota-82 — Load Forecasting & Digital Twin System
### Project Overview Document

---

## 1. What is the Project About / Initiative?

This project is an **AI-powered Electrical Load Forecasting and Digital Twin System** developed for the **Ghana Grid Company (GRIDCo)**, focused specifically on the **Achimota-82 transmission substation** — one of the key nodes in Ghana's national grid.

At its core, the system does three things:

### 1.1 — Forecasts electrical load with machine learning
It ingests high-resolution **15-minute SCADA telemetry** from three power transformers (82T1, 82T3, 82T4) and uses a multi-model ensemble to predict `Community_Load_MW` across multiple horizons — from 15 minutes ahead all the way to 30 days. The forecasting engine includes:
- **LightGBM** (champion model — decomposition residual approach)
- **CNN-BiLSTM** (deep learning — captures short-range temporal dynamics)
- **Physics-Aware LSTM** (Sobolev Trajectory Loss — enforces physical grid constraints)
- **Autoformer & PatchTST** (state-of-the-art transformers for long-range forecasting)
- **Holt-Winters ETS** and **Similar-Day / K-NN** (statistical baselines)

### 1.2 — Maintains a Digital Twin of the substation
The system models not just load, but also physical state variables — **Voltage (kV)** and **Reactive Power (MVAR)** — using physics-informed constraints, so the forecasted future state of the substation is physically plausible, not just statistically likely.

### 1.3 — Provides an operational web dashboard
A full-stack **Next.js + FastAPI** web application gives grid operators a real-time dispatch interface with live SCADA monitoring, forecast visualisation, regime analysis (Standard / Transition / Peak), SHAP explainability, and scenario simulation.

---

## 2. What Major Issue is the Project Seeking to Address?

### The Core Problem: GRIDCo operators forecast load manually — with Excel.

Before this project, load forecasting at Achimota-82 was done using **rule-of-thumb heuristics and historical average lookups in spreadsheets**. This creates several compounding problems:

| Problem | Consequence |
|---|---|
| **No intra-day precision** — Excel cannot model 15-min resolution dynamics | Operators cannot anticipate rapid demand ramps (e.g. air-conditioning surge at 18:00) |
| **No outage-awareness** — raw SCADA data includes grid collapses and sensor failures mixed in with real load | Forecasts trained on dirty data are systematically biased |
| **No uncertainty quantification** — single-point forecasts with no confidence bounds | Operators have no way to distinguish a confident forecast from a guess |
| **No physical consistency** — load predictions are pure statistics, disconnected from voltage or reactive power | Dispatch decisions based purely on MW numbers can violate grid stability constraints |
| **No scalability to multi-day horizons** — Excel averages break down beyond 24 hours | Generation scheduling and fuel procurement decisions are made blind |

The project directly addresses all five of these. Specifically, the **Outage-Aware evaluation framework** (filtering days where daily max load < 25 MW, or where Z-score > 2.5 flags an anomalous regime) is one of the most technically rigorous aspects — it separates forecasting failure *due to model error* from failure *due to the grid being in a collapsed state*, which is a meaningless target for any model to try to predict.

---

## 3. Is the Project Scalable?

**Yes — deliberately so.** The architecture was designed with multi-site, multi-horizon scalability in mind from the beginning.

### Horizontal scaling (more substations)
The entire feature engineering pipeline (`FEATURE_ENGINEERING/config.py`) uses a **column-mapping configuration** that can be re-pointed to any substation's SCADA schema. Adding a new substation is a config change, not a code change. The ML models are trained per-substation, so each node gets a tuned champion model without interference.

### Vertical scaling (longer horizons)
The system already supports **five distinct planning horizons**:
- `T+15min` — for real-time dispatch
- `T+1H` — for hourly unit commitment
- `T+6H` — for intra-day balancing
- `T+24H` — for day-ahead market settlement
- `T+7D / T+30D` — for generation scheduling and fuel procurement

Each horizon uses a **different model architecture** (a design called the "Grand Showdown" benchmark), because no single model dominates across all time scales.

### Infrastructure scalability
- The **backend is FastAPI** (async Python) — can be containerised and load-balanced
- The **feature store** uses Pandas/Parquet — can be migrated to Apache Arrow or a time-series database (InfluxDB, TimescaleDB) for multi-site ingestion
- The **forecast cache** uses horizon-aware keying, so forecast requests for different horizons don't invalidate each other

### Acknowledged limitation
The current dataset covers a **single substation over a ~2-3 year window**. Scaling to the full GRIDCo network (Tema, Pokuase, Aboadze interconnects) would require network-topology-aware models (Graph Neural Networks or spatial correlation matrices), which is not yet implemented.

---

## 4. Do We Have Anything to Add?

There are several high-value extensions the project is positioned to add, in order of maturity:

### Ready to implement (infrastructure exists)
- **Probabilistic dispatch dashboard** — P10/P50/P90 quantile bands from PatchTST are already computed; the frontend needs a "risk envelope" visualisation layer so operators can see the *worst case* scenario alongside the point forecast.
- **Alert thresholds on load ramps** — the SCADA pipeline already detects regime transitions; formalising this into email/SMS alerts for >15 MW/15min ramp events would directly serve dispatch operators.
- **Walk-Forward validation live view** — the benchmark scripts already produce MAE/MAPE per horizon; surfacing this live in the Model Performance tab would close the accuracy feedback loop.

### Medium-term (requires new data)
- **Temperature & weather integration** — the current models use only SCADA load and temporal features. Adding NWP (Numerical Weather Prediction) data from Ghana Meteorological Agency would likely reduce MAPE by 1.5–2.5 percentage points, especially during dry-season peak stress periods.
- **GDP / economic activity proxies** — long-term (30-day) forecasts for generation scheduling could incorporate industrial output or commercial activity data from GSS (Ghana Statistical Service) to capture demand growth trends that historical load averages miss.

### Strategic (longer-term vision)
- **Multi-substation spatial model** — a Graph Neural Network layer over the GRIDCo transmission topology, so a contingency at one node (e.g. forced outage at Tema) automatically propagates adjusted forecasts to upstream nodes like Achimota.
- **Reinforcement Learning for dispatch optimisation** — once the digital twin is validated, the simulated environment can be used to train an RL agent that learns optimal generator dispatch policies, moving from *forecasting load* to *recommending generation mix*.
- **Grid carbon accounting** — pairing load forecasts with Ghana's generation mix (hydro, thermal, imports) to give operators a forward-looking CO₂ intensity signal per dispatch decision — a key metric as Ghana advances its renewable energy commitments.

---

*Document generated: April 2026. Based on system architecture, SCADA dataset analysis, and benchmarking results from the Achimota-82 GRIDCo Load Forecasting Project.*
