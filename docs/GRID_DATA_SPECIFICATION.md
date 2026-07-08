# GRIDCo Master Data Specification: Long-Term Forecasting & Digital Twin Simulatability

This document outlines the mandatory data requirements to transition the GRIDCo load forecasting pipeline from an experimental 15-minute basis to a production-grade, high-accuracy (month-ahead) forecasting and digital twin system.

---

## 1. Long-Term Load Forecasting (LTLF) Requirements
*Goal: Achieve <5% MAPE for month-ahead daily peak predictions.*

### 1.1 Historical Depth
* **Target Volume**: 3–5 years of continuous SCADA history (Minimum: 36 months).
* **Resolution**: 15-minute intervals (integrated to Hourly/Daily for LTLF).
* **Substation Focus**: Nayagina-82 (Primary), Mallam, and Volta links.

### 1.2 Exogenous Features (Multi-Modal)
* **Weather Data**:
    * Temperature (°C), Relative Humidity (%), and Solar Irradiance (W/m²).
    * Source: Ghana Meteorological Agency (GMet) or high-resolution reanalysis (ERA5).
* **Socio-Economic Indices**:
    * National Holidays (Ghana specific).
    * Major Grid Maintenance schedules.
    * Residential vs. Industrial tariff load-shedding regimes.

---

## 2. Digital Twin & Simulatability Requirements
*Goal: Model Voltage (KV) and Reactive Power (MX) with physical consistency.*

### 2.1 Substation Topology (The "Tau" State)
| Feature | Type | Description |
| :--- | :--- | :--- |
| **Breaker Status** | Binary (0/1) | Status of feeders 82AD1, 82NY, 82ZA. |
| **Bus-Coupler** | Binary (0/1) | Indicates if the main busbar is split or solid. |
| **Tap Positions** | Integer (1–17) | On-Load Tap Changer (OLTC) positions for 82T1–T4. |
| **Capacitor Banks**| Binary (0/1) | Status of reactive compensation units. |

### 2.2 Network Physical Parameters
To implement Physics-Informed Neural Networks (PINNs), we require the static constants:
* **Line Impedance**: Resistance ($R$), Reactance ($X$), and Susceptance ($B$) for the transmission corridors.
* **Transformer Nameplate**: Short-circuit impedance ($U_k\%$) and copper/iron loss constants.

---

## 3. Data Quality & Preprocessing Protocols
To prevent model drift and handle sensor decay (as observed in June 2025):

1.  **Life-Sign Filter**: Minimum valid load threshold of **50 MW** for Nayagina-82 (to distinguish outages from sensor failure).
2.  **Telemetry Sanity**: Check for "Frozen Signals" where standard deviation over 1 hour is exactly 0.0.
3.  **Outlier Regime**: Mapping of "System Collapse" events (frequency < 49.0 Hz) to separate training labels.

---

## 4. Summary of Model Horizon Strategy
* **Short-Term (0-6h)**: Recursive LightGBM on 15-minute SCADA.
* **Long-Term (1d-30d)**: Daily Peak Recursion using Expanded Historical Baseline.
* **Digital Twin**: Transformer-based Surrogate mapping Injections $\to$ Voltage/Reactive Power.
