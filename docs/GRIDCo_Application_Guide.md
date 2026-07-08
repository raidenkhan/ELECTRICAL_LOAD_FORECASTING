# GRIDCo Forecast Pro: Application Functional Guide

This document provides a comprehensive overview of the web application's architecture and page-by-step functionality, mapped directly to the operational requirements of the GRIDCo Achimota-82 substation.

---

## 1. Overview Dashboard (Tactical Command)
**Purpose**: The primary "at-a-glance" status for the national load dispatcher.
- **System Overview**: Displays the estimated community load (MW) in real-time, calculated as the sum of T1, T3, and T4 transformer banks.
- **Accuracy Index**: Real-time tracking of the model's R² stability, ensuring the dispatcher knows if the AI is currently "locked on" to the grid's rhythm.
- **Short-Term Load Forecast (STLF)**: A 24-hour rolling chart with P10/P90 confidence bands, allowing engineers to see the expected evening peak before it arrives.
- **Operating Regime Probabilities**: A stacked bar chart showing the statistical likelihood of the grid transitioning from "Standard" to "Peak" or "Transition" regimes throughout the day.

## 2. Digital Twin Dashboard (The Control Room)
**Purpose**: A "What-If" simulation environment for testing grid elasticity.
- **Control Param (Simulation Engine)**: Allows operators to manually adjust the **Simulated Temperature**. This is critical because every 1°C increase in temperature correlates to a ~160 MW load surge due to air-conditioning demand.
- **Scenario Mode**: Toggle between 'Normal', 'Wet', and 'Holiday' profiles to see how different socio-climatic conditions shift the demand curve.
- **Grid Intelligence**: Real-time calculation of the **Available Margin**. If the margin drops below 100 MW, the system triggers a "Critical Margin Warning" and suggests load-shedding mitigation.
- **Generation Unit Monitor**: Displays the active dispatch of units like TICO, Akosombo, and Bui, matching the simulated demand against the total generation capacity.

## 3. Live Monitoring (SCADA Sync)
**Purpose**: High-frequency monitoring of raw SCADA telemetry.
- **Pulse Monitoring**: Tracks individual line flows and transformer telemetry at 15-minute intervals.
- **Inflow Tracking**: Specifically monitors the **NY6ZA Line**, which acts as the "Leading Indicator" for the Achimota substation.
- **System Health**: Displays the "Last Sync" timestamp and SCADA heartbeat, ensuring data integrity for the AI engines.

## 4. Planner Dashboard (Strategic Capacity)
**Purpose**: Long-term resource and fuel planning (1-Week to 30-Day horizons).
- **Weekly Outlook**: A 7-day tactical view highlighting "Critical Peak" days where demand is projected to exceed transformer limits (e.g., >1,600 MW).
- **Monthly Scaling (LTLF)**: Compares current monthly trends against the previous year's average. This identifies the **8% YoY Growth Rate** and helps in bulk fuel procurement strategies.
- **GRIDCo What-If Builder**: A more advanced simulator for planning new industrial connections or major grid maintenance shutdowns.

## 5. Model Performance (Analytics Audit)
**Purpose**: Ensuring the AI remains "Physics-Aware" and accurate.
- **Champion Model Tracking**: Compares the **Physics-Aware LSTM** against the **Similar-Day Heuristic** (GRIDCo's legacy Excel method).
- **Error Heatmaps**: Visualizes which hours of the day (e.g., the 19:00 evening ramp) are seeing the most forecast drift.
- **Ramp Tracking Efficiency**: Metrics showing how well the AI is capturing the "MW/min" gradient of the morning and evening surges.

## 6. Explainability View (XAI / SHAP)
**Purpose**: Building trust with engineers by explaining "Why" the AI made a prediction.
- **Feature Importance**: Breaks down the drivers of a specific forecast. (e.g., "50% of this 1500 MW peak is driven by the 24-hour Load Lag, 20% by current Temperature").
- **SHAP Dependency Plots**: Shows the relationship between individual variables (like Frequency or Oil Temp) and the final load prediction.

## 7. Data Management (Ingestion & Validation)
**Purpose**: The gateway for SCADA data and model retraining.
- **Validation Engine**: Automatically checks uploaded CSVs for "Frozen Signals" (sensor failure) and "Sign Convention Errors" (incorrectly summing generation as load).
- **Regime Labeling**: Identifies and flags "Outage Days" (Dumsor events) so they do not contaminate the training of the normal-regime models.

## 8. Settings & Security
**Purpose**: System configuration and access control.
- **Role-Based Access**: Specialized views for **Operators** (Control Room), **Analysts** (Performance), and **Planners** (Strategy).
- **Threshold Configuration**: Allows engineers to set the MW limits that trigger the "Critical" alerts across the dashboards.
