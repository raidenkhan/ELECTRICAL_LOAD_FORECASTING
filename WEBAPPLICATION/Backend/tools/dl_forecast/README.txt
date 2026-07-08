ECG Load Forecasting -- Deep Learning Training Package
=====================================================

HOW TO USE
----------

1. Install dependencies (GPU PC):
   pip install torch pandas numpy scikit-learn

2. Copy ecg_demand_2018_2026.csv to the same directory as the scripts, or
   adjust --data_path.

3. Run all models:
   python run_cv.py

4. Run specific models:
   python run_cv.py --models lstm transformer

5. Run with custom batch size (adjust to GPU memory):
   python run_cv.py --batch_size 8192

6. Run with batch size tuning (try 4096, 8192, 16384 for GPU):
   python run_cv.py --models lstm tcn --batch_size 8192

MODELS AVAILABLE
----------------
- lstm:       3-layer LSTM, hidden_size=256
- gru:        3-layer GRU, hidden_size=256
- transformer: Encoder-only Transformer, d_model=128, nhead=4, 3 layers
- tcn:        Temporal Convolutional Network, 4 blocks
- dlinear:    Decomposition-Linear from "Are Transformers Effective..."

OUTPUT
------
Results saved to results/ directory:
- results.csv: per-fold, per-model MAE
- per_hour_mae_{model}.csv: 24-hour MAE breakdown (hourly)
- {model}_Fold_*_best.pt: best model checkpoint per fold

GPU REQUIREMENTS
----------------
- At batch_size=4096, any modern GPU (RTX 3060+) works under 8GB VRAM
- batch_size=8192 recommended for RTX 4070+ / A-series
- Mixed precision (FP16) enabled by default
- torch.compile enabled by default for ~30% speedup
- Average training: ~2-5 minutes per fold per model
- Full 6-fold CV for all 5 models: ~30-60 minutes

DATA
----
70,228 hourly rows, 2018-01-01 to 2026-05-01
Columns: date, hour, demand_mw, temperature_c, is_holiday

FOLD STRUCTURE
--------------
Each fold trains on data starting from 2018-01-01
- Fold 1: train to 2019-12-31, test 2020 H1
- Fold 2: train to 2020-12-31, test 2021 H1
- Fold 3: train to 2021-12-31, test 2022 H1
- Fold 4: train to 2022-12-31, test 2023 H1
- Fold 5: train to 2023-12-31, test 2024 H1
- Fold 6: train to 2024-12-31, test 2025 H1

ARCHITECTURE NOTES
------------------
Input:   (batch, 168, 9) - 7 days of 9 features
Output:  (batch, 24)     - next 24 hours of demand

Features:
  0: demand_mw (z-score)
  1: hour_sin
  2: hour_cos
  3: dow_sin
  4: dow_cos
  5: month_sin
  6: month_cos
  7: temperature_c (z-score)
  8: is_holiday

The DLinear model assumes demand_mw is at feature index 0.
All other models treat features uniformly.
