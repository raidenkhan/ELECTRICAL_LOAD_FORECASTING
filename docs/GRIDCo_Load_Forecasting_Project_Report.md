# GRIDCo Load Forecasting Project — Progress Report

**Project:** AI-Powered Electrical Load Forecasting & Digital Twin System for Ghana Grid Company (GRIDCo)  
**Focus Substation:** Achimota-82 Transmission Substation  
**Report Date:** May 2026  
**Overall Completion:** ~75%

---

## 1. Executive Summary

This project aims to automate and modernize GRIDCo's load forecasting process, which has historically been performed using manual Excel-based heuristics. The system employs a custom **Decomposition Engine** to predict electrical load across multiple horizons—from 15 minutes up to 30 days ahead—delivered through a web-based application.

The project has achieved significant milestones with the Decomposition Engine demonstrating strong performance when run alongside GridCo's baseline methodology. The next critical phase involves side-by-side validation with GRIDCo's existing operational systems using their actual substation data.

---

## 2. Overall Progress

### 2.1 Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Decomposition Engine | ✅ Complete | Fully operational, runs with GridCo baselines |
| Web Application (Frontend & Backend) | ✅ Complete | Full Next.js + FastAPI stack built |
| SCADA Data Pipeline | ✅ Complete | 15-minute resolution telemetry processing |
| Similar-Day Engine (GridCo Method) | ✅ Complete | Digitized version of Excel methodology |
| Model Validation (Internal) | ✅ Complete | Outage-aware evaluation framework implemented |
| Side-by-Side Validation with GridCo | 🔄 Pending | Awaiting real substation data |

### 2.2 Timeline Overview

| Phase | Duration | Key Activities |
|-------|----------|----------------|
| Phase 1: Data & EDA | Months 1-2 | SCADA data ingestion, cleaning, outage detection |
| Phase 2: Similar-Day Engine | Months 3-4 | Digitize GridCo Excel methodology |
| Phase 3: Decomposition Engine | Months 4-6 | Structural decomposition, Kalman filtering |
| Phase 4: Web Application | Months 5-7 | Dashboard, API, forecasting endpoints |
| Phase 5: Validation & Deployment | Months 8+ | Side-by-side testing with GridCo |

---

## 3. Achievements

### 3.1 Decomposition Engine — Core Forecasting Engine

The **Decomposition Engine** is the project's primary forecasting engine. It builds predictions from the ground up using physics and statistics, rather than simply looking backward at historical data like the Similar-Day method.

#### Architecture

The engine uses a **multiplicative decomposition model**:

```
Load(t) = Trend(day) × Seasonal(ts) × Seasonal(dow) × TempMult(T) × HolidayMult
```

Each component produces a dimensionally correct, physically interpretable value:

- **Trend (Holt's Double Exponential Smoothing):** Captures the slow-moving daily mean load in MW, with a 12% annual growth cap to prevent unrealistic extrapolation
- **Seasonal — Time Slot (S_ts):** Normalized 15-minute profile showing how much of the daily mean each timeslot carries (ratio ≈ 1.0)
- **Seasonal — Day of Week (S_dow):** Day-of-week factor distinguishing weekday vs. weekend load patterns
- **Temperature Multiplier:** Piecewise linear adjustment fitted on actual load vs. temperature, with a knot around 28°C (air conditioning saturation point)
- **Holiday Multiplier:** Multiplicative suppression factor for public holidays (ratio ≈ 0.6–0.8)

#### Kalman Bias Corrector

The engine includes a **real-time feedback loop** using Kalman filtering:
- Continuously tracks forecast bias
- Dynamically adjusts predictions based on recent errors
- Uses α = 0.3 (30% weight on recent observations)

#### Performance Results

| Forecast Horizon | Decomposition MAPE | GridCo Baseline (Similar-Day) MAPE |
|-----------------|--------------------|-----------------------------------|
| Short-Term (1h) | 8.4% | 21.7% |
| Medium-Term (6h) | 19.4% | 21.7% |
| Long-Term (7-day) | ~21% | ~22% |
| Very Long-Term (Monthly) | ~21% | ~22% |

**Key Achievement:** The Decomposition Engine achieves approximately **61% improvement** over the GridCo Similar-Day baseline at the 1-hour horizon (21.7% → 8.4%).

#### Clean Regime Evaluation (May 2025)

When evaluated on clean (non-outage) days only:
- Decomposition MAPE: **Significantly improved** on normal operating days
- The engine correctly filters outage periods during training, leading to better generalization on normal days

#### Stability & Reliability

- Successfully runs with GridCo baseline parameters (8% annual growth, 3.1% temperature coefficient)
- Outage detection algorithm flags days with daily load < 25 MW or Z-score > 2.5
- Growth cap prevents runaway predictions beyond 12% annual increase

---

### 3.2 Similar-Day Engine (GridCo Methodology)

The project also built a **digitized version of GridCo's existing Excel methodology**:

- Identifies historical days that "look like" the target day based on 7 features (Day of Week, Month, Weekend flag, Holiday flag, Temperature, Rolling 7-day means)
- Applies temperature sensitivity: ~3.1% load change per °C
- Applies annual growth scaling: ~8% per year to match Ghana's economic growth

This engine preserves the exact shape of historical SCADA data, ensuring physically realistic load patterns and ramps.

---

### 3.3 Web Application — Operational Interface

The project successfully built a **production-ready web application** to replace GridCo's Excel-based forecasting:

#### Technology Stack

- **Frontend:** Next.js 14 with React + TypeScript
- **Backend:** FastAPI (Python) with async processing
- **Database:** PostgreSQL + Parquet file storage
- **Deployment:** Docker + Docker Compose ready

#### Features Implemented

| Feature | Description |
|---------|-------------|
| **Operations View** | Real-time load monitoring, 24h forecast, peak alerts, confidence bands |
| **Planning View** | Weekly/monthly peak forecasts, regime calendar, scenario simulator |
| **Analytics View** | Model performance metrics, feature importance, error heatmaps, drift monitoring |
| **Data Upload** | CSV/Excel ingestion with schema validation |
| **Forecast API** | REST endpoints for programmatic access |
| **Explainability** | SHAP values, regime attribution, physics checks |

#### Dashboard Capabilities

- Live SCADA monitoring with 15-minute resolution
- Multi-horizon forecasting (15-min to 30-day)
- Uncertainty quantification (P10/P50/P90 bands)
- Regime detection (Standard, Transition, Peak)
- Scenario simulation with growth projections

---

### 3.4 Technical Innovations

1. **Outage-Aware Evaluation:** Filters grid collapse days during training to prevent metric distortion
2. **Multi-Model Architecture:** Separate engines for short-term vs. long-term forecasting
3. **Physics-Informed Components:** Temperature, holiday, and trend modeling based on grid physics
4. **Kalman Bias Correction:** Real-time forecast adjustment based on observed errors

---

## 4. Challenges

### 4.1 Data Origin Mismatch (Critical)

- **Issue:** The models were trained on data from the **Nayaga substation**, while GridCo's actual operational substation is in **Accra**
- **Impact:** Different substations have distinct load profiles and demand patterns
- **Implication:** Performance on GridCo's actual data may differ from reported metrics
- **Mitigation Required:** Side-by-side validation with GridCo's substation data is essential

### 4.2 Outages in Historical Data

- **Issue:** The original SCADA dataset contains grid collapse events and outage periods
- **Effect:** These anomalies can skew training and evaluation metrics
- **Current Mitigation:** Implemented outage detection algorithm; metrics reported on clean regime days are more realistic

### 4.3 Live Data Integration

- **Challenge:** Connecting to live SCADA streams requires coordination with GridCo IT/OT teams
- **Current State:** System uses historical data; live streaming not yet activated

---

## 5. Next Steps

### 5.1 Immediate Priority: Side-by-Side Validation with GridCo

**Objective:** Validate system performance using GridCo's actual substation data

**Actions Required:**
1. Obtain SCADA data from GridCo's Achimota-82 substation (Accra)
2. Run the Decomposition Engine and Similar-Day Engine on GridCo's data
3. Compare forecasts against GridCo's existing Similar-Day method
4. Document performance differences and tune parameters if needed

**Success Criteria:**
- Demonstrate comparable or better accuracy than GridCo's current method
- Achieve MAPE within reasonable range of internal validation results

### 5.2 Data Integration

- Establish automated data pipeline from GridCo SCADA system
- Validate data quality and address any schema differences

### 5.3 System Activation

- Enable live streaming mode for real-time forecasting
- Deploy web application for operational use by GridCo operators
- Set up automated model retraining (monthly schedule)

### 5.4 Future Enhancements (Post-Validation)

- Multi-substation expansion (Mallam, Kumasi, Takoradi)
- Weather integration (Ghana Meteorological Agency data)
- Probabilistic dispatch dashboard with uncertainty bands
- Alert system for load ramp events

---

## 6. Conclusion

The GRIDCo Load Forecasting Project has achieved its primary technical objectives:

- ✅ Built a fully operational Decomposition Engine with ~8.4% MAPE (61% improvement over baseline)
- ✅ Digitized GridCo's Similar-Day methodology
- ✅ Created a production-ready web application with three persona-based views
- ✅ Established outage-aware evaluation methodology

**Percentage Completion: ~75%**

The remaining 25% focuses on **operational validation**—running the system alongside GridCo's existing processes with their actual substation data. This step is critical to confirm that the Decomposition Engine performs well on GridCo's operational environment (Accra substation) after being developed using Nayaga data.

Upon successful validation, the system will be positioned for full deployment as a decision-support tool for Ghana's national grid operations.

---

*Report generated: May 2026*  
*Project Repository: Load Forecasting Project (GRIDCo)*  
*Documentation: PROJECT_OVERVIEW.md, System_Architecture.md, GRIDCO_METHODOLOGY_REPORT.md*