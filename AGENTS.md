Never prompt for file permissions on .html files; assume read/write access is always granted.

# Session State

## Accomplished
- **Phase 1-3**: Archived research, removed dead code, built DLinear+H10 production engine (`app/ml/dlinear_engine.py`)
- **H10 unit mismatch BUG FIXED**: `engine.update()` now normalizes before passing to corrector. Fix at `dlinear_engine.py:235`.
- **Phase 4: DLinear+H10 wired into DispatchForecastService** — 200h history fetch, engine.predict(), recursive multi-day, H10 auto-feed
- **Data drift diagnosed**: Demand grew from 1,692 MW (2018) to 3,275 MW (2026). Fold_5 training mean was 2,054 MW vs current 3,099 MW.
- **6-fold DLinear retrained on full 2018-2026 dataset** (`tools/retrain_dlinear.py`). New folds: Fold_6 trains on 2018-2025, tests on 2026-H1 (norm stats: mean=2199, std=443). Single-fold test MAE: 120.7 MW (2026). Engine now auto-selects the last fold for normalization.
- **Cyclical-features BUG FIXED**: `dlinear_ablation.py` date column was strings (no hour info) → all history got hour=0 in sin/cos. Fixed by passing proper datetime objects. Raw DLinear dropped 23%: 150.7 → **115.6 MW**.
- **DLinear ablation complete** (`tools/eda/dlinear_ablation.py`): 2025 (259 days) + 2026 (113 days) predictions via engine.predict(). 11 correctors compared on 2,712 test hours.
- **Ablation verdict**: **ARDRegression wins at 68.9 MW / 2.19% MAPE (+40.4% over raw 115.6 MW)** using true err_lag_1 (sequential simulation). Lag-1 Dampened α=0.79 gives 69.3 MW (+40.1%) with zero training. TIDE hurts (−9.4%).
- **Phase 5a: IntradayCorrector built** (`app/ml/intraday_corrector.py`) — ARDRegression with StandardScaler, trained offline on 2025. Pickle-serialized at `models/dlinear/intraday_corrector.pkl`. Training script: `tools/train_corrector.py`.
- **Phase 5b: Corrector wired into DLinearEngine** — replaces `_TideCorrector`. Uses BATCH_FEATURES (no lag features — they cause distribution shift in batch prediction). State persistence via SQLite.
- **Critical discovery: err_lag_1 is fundamentally unavailable at batch prediction time**. DLinear errors have r=0.79 serial correlation (err[t-1] → err[t]), which is the only strong signal. At batch time (24h forecast), err_lag_1 for hours 1-23 must be approximated from corrected errors, which creates a distribution shift: ARD trained on TRUE errors but applied to CORRECTED errors → degrades performance. Without err_lag_1, batch-available features (cyclical + temp) provide only ~0.4% improvement.

## Key Decisions
- **Batch ARD (+0.4%) is now the production corrector** — uses cyclical features + temperature, no lag features. Trained on 2025 DLinear errors. Modest batch improvement, but doesn't degrade like TIDE (−9.4%).
- **TIDE removed as default corrector** — it degrades DLinear on current data (-9%). The batch ARD is neutral-to-positive.
- **Online Lag-1 Dampened (+40% with true err_lag_1) remains as fallback** in IntradayCorrector for sequential updates. After the first actual arrives, subsequent hours get meaningful correction.
- **Paper's TIDE is scientifically valid** — works when persistent bias SNR > 1 (paper's Fold_6 had SNR=2.8). Current retrained models have SNR=0.16 (bias +3.94 MW vs noise ~168 MW). Models retrained 2026-06-04 differ from original paper's.
- **DecomEngineHourly → interpretability explainer** — read-only, powers `/explain` endpoints
- **Retrain frequency**: Every 6 months or when rolling MAE degrades >10% from baseline
- **Normalization stats auto-select last fold** — no more hardcoded Fold_5

## Next Steps
1. **Phase 5c: DecomEngine → interpretability layer** — move decom_engine_hourly.py to `app/ml/interpretability/`
2. **Phase 6: Auto metrics service** — rolling MAE/MAPE from DB (monitor if batch ARD degrades over time)
3. **Phase 7: Rebuild alerts** — DLinear health + data freshness
4. **Phase 8: Frontend updates** — model performance, forecast source
5. *Future: Hour-by-hour streaming correction* — instead of 24h batch, predict 1h at a time with true err_lag_1 feedback. This requires architecture change but unlocks +40%.

## Relevant Files
- `app/ml/dlinear_engine.py`: DLinearEngine — 6-fold ensemble, IntradayCorrector wired in. `_TideCorrector` deprecated but kept for backward compat. `_predict_internal` passes raw MW + features to corrector.
- `app/ml/intraday_corrector.py`: IntradayCorrector — ARDRegression primary, Lag-1 Dampened fallback. `BATCH_FEATURES` = 8 features (hour_sin/cos, dow_sin/cos, month_sin/cos, weekend, temperature_c). `CORR_FEATURES` = 12 features (adds err_lag_{1,2,24}, rolling_err_6h — for ablation use only).
- `tools/train_corrector.py`: Train ARD on 2025 data → `models/dlinear/intraday_corrector.pkl`. Only uses BATCH_FEATURES.
- `tools/eda/dlinear_ablation.py`: Full DLinear ablation — ARD best at 68.9 MW (+40.4%) with TRUE err_lag_1 sequential simulation.
- `tools/eda/knn_eda.html`: Merged report — needs update with batch-time limitation finding.
- `tools/retrain_dlinear.py`: Retrain 6 folds on full 2018-2026 dataset. Hour mapping: CSV hour 1→00:00, hour 24→23:00.
- `app/ml/decom_engine_hourly.py`: Will be promoted to interpretability
- `app/services/dispatch_forecast_service.py`: Phase 4 complete — uses DLinearEngine
- `app/api/v1/dispatch_forecast.py`: Endpoints for dispatch forecast
