---
marp: true
theme: default
paginate: true
header: 'GRIDCo Load Forecasting: Audit & Automation'
footer: 'Methodological Transparency Report - April 2026'
backgroundColor: #ffffff
---

# GRIDCo Load Forecasting: 
## Audit of Methodological Flaws & The Path to True Automation

**Focus**: Exposing "Wrong Assumptions" and Implementing Physics-Aware Solutions
**Based on**: `output/` Directory Analysis

---

# 1. The Starting Point: A Methodological Audit
### We audited the existing AI claims and found 10 critical flaws:
*   **The "Independent Line" Myth**: Lines were modeled as separate tracks, ignoring the physical coupling of the grid (Claim 2).
*   **The Frequency Fallacy**: Treating frequency as a per-line variable when it is a system-wide constant (Claim 1).
*   **15-Minute Masking**: Using 15-minute averaged data to claim "fast fault" detection—a physical impossibility (Claim 4).
*   **The "Perfect" Data Trap**: 17 months of zero missing values suggests heavy, potentially biased interpolation (Claim 5).

---

# 2. Exposing "Wrong Assumptions" (The Dishonesties)
### To make AI look superior, previous tests used "Dishonest Metrics":

1.  **Perfect Weather Foresight (Data Leakage)**:
    *   **The Flaw**: Feeding the *actual* temperature of the test day into a 30-day forecast.
    *   **The Reality**: You don't know the temperature 30 days in advance.
    *   **The Impact**: When we removed this "clairvoyance," AI error jumped from **27% to 31.5%**.

2.  **The "Strawman" Baseline**:
    *   **The Flaw**: Comparing AI against a "dumb" 365-day shift.
    *   **The Reality**: This baseline predicts Sundays as Mondays, making it look much worse than GRIDCo's actual manual methods.

---

# 3. Automation: Solving the "Slow Reaction" Problem
### We transitioned from manual Excel heuristics to automated Deep Learning:

*   **Original Heuristic**: Manual YoY (Year-over-Year) multipliers + "Gut feeling."
*   **Our Automation**: **Physics-Aware LSTM** using Sobolev Trajectory Loss.
*   **The Breakthrough**: Instead of just matching numbers, the AI now tracks the **Slope (MW/min)**.
*   **Result**: Eliminated the "Lag" where AI would miss sudden industrial ramps.

---

# 4. Proving Facts: The Grand Showdown
### [Graph: output/grand_showdown/mape_showdown.png]
*   **Observation**: Accuracy naturally decays as the "Blindness Window" grows.
*   **The "Honest" Win**: Even without data leakage, the **Physics-Aware LSTM** (27.12% MAPE) beats the Legacy Baseline (30.68%) at the 30-day horizon.
*   **Fact**: Automation provides a **3.5% absolute accuracy gain** for long-term strategic planning.

---

# 5. Difficulties in Metrics: The Outage Effect
### [Graph: output/claim_4_analysis.png]
*   **The Difficulty**: Grid outages (like June 2025 "Dumsor") cause load to drop to ~11 MW.
*   **The Metric Bias**: If the AI predicts the *true demand* (80 MW) during a blackout (0 MW), the math says the AI is "wrong."
*   **Our Correction**: We use a **Z-Score Regime Detector** to isolate "Normal Grid Days" from outages, ensuring our 17.5% accuracy reflects reality, not noise.

---

# 6. Why We Need More Data
### To reach <5% MAPE, we must move beyond current constraints:

1.  **Climate Drift**: We need 3-5 years of history to understand how urbanization in Ghana is shifting the "Baseload Floor."
2.  **Thermal Inertia**: Current models guess based on Oil Temp. We need direct SCADA feeds of **Ambient Temperature** and **Humidity** to stop "guessing" the air-conditioning load.
3.  **Industrial Meta-Data**: Knowing when a major factory is scheduled for maintenance would turn an "unpredictable drop" into a "planned event."

---

# 7. Summary: Operational Recommendations
### Based on the `MODEL_COMPARISON_REPORT.md`:

*   **Intra-Day (<24h)**: Use **CNN-BiLSTM**. It is the most robust to 15-minute "jitter."
*   **Long-Term (>7 days)**: Use **Physics-Aware LSTM**. It masters the daily double-peak cycle and temperature gradients.
*   **Safety First**: Use **P90 Quantile forecasts** to ensure transformer limits are never breached during unexpected surges.

**Automation is not just about better math; it is about honest, physics-grounded engineering.**
