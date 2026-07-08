# GRIDCo Load Forecasting & Digital Twin System

## Project Overview
This project is a comprehensive system for electrical load forecasting and digital twin simulation for the GRIDCo (Ghana Grid Company) network, specifically focused on the **Achimota-82 substation**. It handles high-resolution (15-minute) SCADA data to predict load patterns across various horizons (from 1 hour to 30 days) while maintaining physical consistency and resilience to grid outages.

### Key Technologies
- **Data Processing**: Python, Pandas, NumPy, Scikit-learn.
- **Machine Learning**: LightGBM, XGBoost, K-Nearest Neighbors (Similar-Day method).
- **Deep Learning**: PyTorch, CNN-BiLSTM, Physics-Aware LSTM (Sobolev Trajectory Loss), Autoformer, PatchTST, DLinear.
- **Statistical Models**: Holt-Winters ETS.
- **Web Application**: Full-stack architecture with a dedicated `Backend` and `frontend`.
- **Visualization**: Matplotlib, Seaborn for analytics; React/Web for the dashboard.

### Core Architecture
1.  **Feature Engineering**: `FEATURE_ENGINEERING/` - Creates 20+ features including cyclic temporal encoding, multi-scale lags (15m, 1h, 24h, 7d), and rolling statistics.
2.  **Modeling Engine**: `MODEL_BUILDING/` - Houses traditional ML, deep learning, and state-of-the-art (SOTA) time-series transformers.
3.  **Digital Twin**: Models physical parameters like Voltage (KV) and Reactive Power (MX) using physics-informed constraints.
4.  **Evaluation Pipeline**: Implements "Outage-Aware" metrics to separate normal grid behavior from system collapses or sensor failures.

---

## Building and Running

### Data Preparation
To generate the engineered features from raw SCADA data:
```bash
python FEATURE_ENGINEERING/feature_engineering_pipeline.py
```

### Execution Scripts
The project uses several main entry points for different forecasting tasks:

- **Day-Ahead Forecasting (15-min resolution)**:
  ```bash
  python run_proper_forecast.py --models lgbm,simday,hw
  ```
- **Comparative Ablation Study (Grand Showdown)**:
  Runs CNN-BiLSTM, Physics-Aware LSTM, and G-KNN across multiple horizons (1h, 6h, 1d, 7d, 30d).
  ```bash
  python run_grand_showdown.py
  ```
- **Physics-Aware Forecasting**:
  ```bash
  python run_physics_aware_forecast.py
  ```
- **SOTA Transformer Experiments**:
  Located in `MODEL_BUILDING/sota/`, specifically using Autoformer and PatchTST for long-term forecasting.

### Web Application
- **Backend**: Navigate to `WEBAPPLICATION/Backend/` and follow the local setup instructions.
- **Frontend**: Navigate to `WEBAPPLICATION/frontend/` and run `npm install && npm start`.

---

## Development Conventions

### Data Handling
- **Sampling**: Primary resolution is 15-minute intervals (96 steps/day).
- **Outage Filtering**: Daily max load < 25 MW or Z-score > 2.5 is used to flag anomalous regimes.
- **Target Variable**: `Community_Load_MW`, calculated as the sum of T1 (clipped to >=0), T3, and T4 transformer loads.

### Modeling Principles
- **No Data Leakage**: All lag and rolling features must use `.shift()` to ensure they only use past information.
- **Validation**: Use chronological time-series splits (Walk-Forward Validation); **DO NOT shuffle data**.
- **Physical Consistency**: Models (especially DL) should respect temperature gradients and line impedance constraints where possible.

### Key Files
- `GRID_DATA_SPECIFICATION.md`: Mandatory requirements for production-grade forecasting.
- `FEATURE_ENGINEERING/config.py`: Global constants and column mappings.
- `run_grand_showdown.py`: The definitive benchmark script for all architectures.
- `MODEL_BUILDING/Model_Building_Report.md`: Detailed analysis of model performance and methodology.
