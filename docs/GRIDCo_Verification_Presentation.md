---
marp: true
theme: default
paginate: true
header: 'GRIDCo AI Project Verification'
footer: 'Data Analysis & Methodology Assessment'
---

# GRIDCo Dataset Verification
## Methodological Assessment of AI Claims

**Objective**: Rigorously verify the methodology proposed in the AI monitoring project using the 15-minute GRIDCo SCADA dataset.
**Result**: 8 severe methodological flaws verified, 2 claims refuted.
**Assessment**: The current methodology is fundamentally UNSOUND for real-world grid deployment.

---

## Overview: Table of Methodological Flaws

Before detailing the evidence, here is a summary of the 10 core methodological flaws identified in the project:

| # | The Claim / Methodology | The Reality (Our Finding) | Severity |
|---|---|---|---|
| **1** | AI predicts Frequency per-line | Frequency is system-wide, identical everywhere. | High |
| **2** | Lines are modeled independently | Lines are highly physically coupled. | High |
| **3** | Datapoints are organic | 100% of data is artificially interpolated to 15-mins. | High |
| **4** | Detects fast faults/anomalies | 15-min averaging erases all fast fault signatures. | Critical |
| **5** | Data has zero missing values | Implausible for 17 months; heavily pre-cleaned. | Major |
| **6** | No fault data exists to train on | We found 2,256 abnormal voltage spikes in the data. | Critical |
| **7** | Uses an 8-hour prediction window | Misses the 24-hour daily peak load cycle entirely. | Major |
| **8** | Models raw numbers | Ignores basic electrical constraints (like Power Factor).| Major |
| **9** | Models "explode" after 5 hours | Accuracy degrades normally, it doesn't catastrophically fail. | Critical |
| **10** | Solves SAIDI/SAIFI blackouts | High-voltage models don't fix neighborhood power lines. | Critical |

---

## 1. Frequency is System-Wide 
**The Flaw**: The paper treats "Frequency" as if every transmission line has its own unique frequency to predict.

**The Reality**: 
Frequency is the heartbeat of the entire national grid—it is exactly ~50Hz everywhere at the exact same time. 

**Analogy**: 
Trying to have an AI predict frequency for *one specific line* is like trying to predict the weather specifically for your driveway instead of your entire city. It is physically inaccurate to predict it locally.

![Claim 1](output/claim_1_analysis.png)

---

## 2. Topological Connection
**The Flaw**: The paper treats the 5 transmission lines aetely independent, separate tracks.

**The Reality**: 
Electricity flows like water in a network of connected pipes. If a "valve" closes on one line, the pressure (power) immediately shifts to the others. Our analysis proves these 5 lines are highly connected. Modulating them in isolation violates the basic laws of physics.

![Claim 2](output/claim_2_analysis.png)

---

## 3 & 4. 15-Minute Data Masking (The Biggest Flaw)
**The Flaw**: The project claims to detect "faults" and "anomalies" using data that was averaged and recorded every 15 minutes.

**The Reality**: 
A real grid fault—like a lightning strike, a short circuit, or a frequency drop—happens in a fraction of a second. Because this data averages everything out every 15 minutes, those fast, dangerous faults are completely erased from the math.

**Analogy**: 
Imagine taking a photograph of a highway with a 15-minute exposure time. A speeding car driving by will be completely invisible. You cannot train an AI to detect "fast faults" using "slow" 15-minute averaged data.

![Claim 4](output/claim_4_analysis.png)

---

## 5. Suspicious "No Missing Values"
**The Flaw**: The raw sensor dataset had exactly zero missing values over 17 entire months.

**The Reality**: 
In the real world, sensors drop offline, communications fail, and data gets lost. A "perfect" 17-month dataset means the authors heavily guessed (interpolated) the missing parts. 

**Analogy**:
If we train an AI to learn from guessed, synthetic data, the AI learns the human guesses, not the actual reality of the power grid.

![Claim 5](output/claim_5_analysis.png)

---

## 6. Labeled Fault Data
**The Flaw**: The paper implies real fault data is scarce and hard to detect.

**The Reality**: 
When we audited the raw data limits, we actually found over 2,000 distinct moments where the voltage dangerously spiked or dropped beyond the safe 161kV tolerance. There *is* sufficient data to identify real disturbances, but the paper's methodology simply missed them.

![Claim 6](output/claim_6_analysis.png)

---

## 7. The 8-Hour Forecast Horizon
**The Flaw**: The AI tries to predict grid behavior 8 hours into the future.

**The Reality**: 
Grid demand operates on a rigid 24-hour cycle—people wake up, go to work, come home, and turn on appliances. An 8-hour window is completely operationally misaligned. 

**Analogy**: 
It's like trying to predict 5 PM rush-hour traffic using data from 9 AM. The AI is completely blind to the evening peak load. It needs a 24-hour view to "see" the evening peak coming.

![Claim 7](output/claim_7_analysis.png)

---

## 8. Missing Physics Guidelines
**The Flaw**: The raw data fed to the AI lacks electrical thermodynamic limits.

**The Reality**: 
The AI is just looking at raw math numbers. Because the authors didn't program in basic electrical constraints (like Power Factor or maximum thermal line capacity), the AI might confidently predict a power flow that would literally melt the physical wires in real life. It doesn't know the rules of electricity.

![Claim 8](output/claim_8_analysis.png)

---

## 9. Overfitting Error Claims
**The Flaw**: The paper claims that basic forecasting models break down completely and explode with errors after 5 hours (20 steps).

**The Reality**: 
We ran our own baseline test. The AI doesn't completely 'explode' with errors after 5 hours as they claimed. The accuracy naturally degrades over time, which is completely normal for any weather or financial forecast, but it doesn't catastrophically fail like they reported.

![Claim 9](output/claim_9_analysis.png)

---

## 10. SAIDI / SAIFI Claims (The Business Flaw)
**The Flaw**: The paper claims this AI will reduce customer blackout metrics (SAIDI/SAIFI) by monitoring these 161kV transmission lines.

**The Reality**: 
161kV lines are the giant bulk "highways" of electricity. However, over 90% of household blackouts happen at the local neighborhood level (e.g., a tree falling on a local distribution street pole). 

**Analogy**:
Improving forecasting on the giant interstate highways has virtually zero impact on the metric that measures whether a local neighborhood street is blocked.

![Claim 10](output/claim_10_analysis.png)

---

# The Way Forward
## Architecting a Valid AI Model for this Dataset

---

## 1. Pivot the Core Objective
**From Fault Detection to Macro-Load Forecasting**

- **Why?** Since 15-minute interpolation permanently erased "fast" fault signatures, anomaly detection is mathematically impossible here.
- **The Pivot**: We must pivot strictly to predicting **Active Power (MW) Demand** and **Voltage Stability**. We should predict daily electricity demand for dispatch planning, where smooth 15-minute trends are actually very beneficial.

---

## 2. Implement Spatio-Temporal AI Architectures
**Leveraging the Connected Grid**

- **Why?** As proven, the 5 lines are highly coupled. Modeling them in isolation is like trying to balance a scale by only looking at one side.
- **The Architecture**: 
  - Use a **Spatio-Temporal Graph Convolutional Network (STGCN)**. 
  - This sounds complex, but it simply means we tell the AI to look at all 5 lines *at the exact same time*. It forces the AI to learn how the electricity shifts across the network from one line to another.

---

## 3. Align Horizons & Engineer Physics
**Fixing the Blindspots**

- **Change the Window**: Shift the AI's output horizon from 8 hours to **24 hours (96 steps)**. This ensures the AI can accurately forecast the critical evening peak load.
- **Enforce Physics**: Calculate the implicit Power Factor (MW/MVA) and feed it to the model. This acts as a "guardrail" to prevent the AI from predicting physically impossible power flows.
- **Global Constraints**: Treat frequency as a grid-wide law, not something to predict per-line.

---

## 4. Give the AI a Calendar 
**Injecting the Grid's Rhythms**

- The grid operates on human schedules. Right now, the AI doesn't know what day it is. We must inject:
  - **Time-of-Day**: So it knows morning vs. night.
  - **Day-of-Week**: So it knows high-demand weekdays vs. low-demand weekends.
  - **Seasonality**: So it accounts for Ghana's dry vs. rainy season load variations (like higher air conditioning usage).
