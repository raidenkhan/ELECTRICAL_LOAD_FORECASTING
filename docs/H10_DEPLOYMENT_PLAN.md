# H10 Adaptive Level Correction — Deployment Plan

**Model:** DLinear (40K params, 6-fold CV) + AdaptiveLevelCorrection(α=0.3, window=48h)
**Performance:** 67 MW / 2.8% MAPE D+1 — beats WT+DOW by 31 MW, plain DLinear by 24 MW
**Cost:** Zero training. ~20 lines Python. ~0.001s per inference.

---

## 1. Production Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  SCADA Feed  │────▶│  Data Ingest     │────▶│  Feature Pipeline  │
│  (hourly)    │     │  (validate/store) │     │  (hour/dow/month   │
└─────────────┘     └──────────────────┘     │  sin/cos, temp)   │
                                              └────────┬──────────┘
                                                       │
                                                       ▼
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  WT+DOW      │◀────│  Engine Selector  │◀────│  DLinear (fold     │
│  (fallback)  │     │  (routing logic)  │     │  ensemble avg)     │
└─────────────┘     └──────────────────┘     └────────┬──────────┘
                                                       │
                                                       ▼
                                              ┌───────────────────┐
                                              │  H10 Bias          │
                                              │  Corrector         │
                                              │  (online, stateful)│
                                              └────────┬──────────┘
                                                       │
                                                       ▼
                                              ┌───────────────────┐
                                              │  Forecast Output   │
                                              │  (24h × 7 horizons)│
                                              └───────────────────┘
```

### 1.1 Integration Points

**A) DLinear inference service**
- Location: `Backend/app/ml/dlinear_engine.py` (to be created)
- Loads all 6 fold checkpoints from `models/dlinear/`
- Computes ensemble mean: raw_pred = mean(fold_1..fold_6 prediction)
- CPU inference: ~0.6s for all 6 folds (6 × 0.1s)
- Single-file, no external model server needed

**B) H10 Corrector (stateful)**
- Location: inline within `dlinear_engine.py` or as `Backend/app/ml/h10_corrector.py`
- State: `error_buffer` (last 48h of pred/actual pairs), `_ema_bias` (24,)
- Persistence: serialize to Redis or SQLite on every SCADA update
- Cold start: pre-populate buffer with 7 days of WT+DOW errors before activating

**C) Engine selector**
- Routes requests to WT+DOW (stable baseline) or DLinear+H10 (best accuracy)
- Rule: If SCADA data is fresh (<2h old), use DLinear+H10; else fall back to WT+DOW
- Allow manual override via API query param `engine=dlinear|wtdow`

**D) Bias state serialization**
```python
# On every SCADA actual arrival:
corrector.update(raw_pred, actual)
redis.set("h10:bias_state", pickle.dumps(corrector))
redis.set("h10:last_update", datetime.utcnow().isoformat())

# On server restart:
state = redis.get("h10:bias_state")
if state:
    corrector = pickle.loads(state)
```

---

## 2. Rollout Strategy

### Phase 0: Shadow Mode (1 week)
- Run DLinear+H10 in parallel with WT+DOW
- Log both predictions, serve WT+DOW to users
- Verify no silent failures (NaN, extreme spikes)
- Collect hourly MAE comparison

### Phase 1: Canary (2 weeks)
- Route 10% of forecast requests to DLinear+H10
- Monitor: error rate, latency, p99 consistency
- Auto-revert if MAE > 115% of WT+DOW baseline

### Phase 2: 50/50 (2 weeks)
- Route 50% of requests to each engine
- Statistical comparison (paired t-test on daily MAE)
- If DLinear+H10 is consistently better → Phase 3

### Phase 3: Full rollout
- DLinear+H10 is primary; WT+DOW is automatic fallback
- Monitoring runs continuously

### Rollback
- API parameter `?engine=wtdow` forces WT+DOW
- If H10 bias diverges (>3σ from historical), auto-fallback
- Operator can toggle via dashboard button

---

## 3. Monitoring & Alerting

### 3.1 Metrics to track

| Metric | Source | Threshold | Action |
|--------|--------|-----------|--------|
| H10 MAE (24h rolling) | Prediction vs SCADA | > 98 MW (WT+DOW baseline) | Auto-fallback to WT+DOW |
| H10 bias magnitude | `get_bias()` | > 150 MW any hour | Clamp bias, log warning |
| Bias drift (7d rolling) | `_ema_bias` history | > 50 MW/hour/week | Flag for retraining |
| SCADA staleness | Last actual timestamp | > 4h | Freeze bias, use last known |
| NaN/Inf predictions | Model output | Any occurrence | Immediate fallback |
| Latency p99 | Inference timing | > 5s | Scale or optimize |

### 3.2 Drift Detection
- Use existing `drift_detector.py` to monitor prediction error distribution
- If 7-day rolling MAE exceeds 110% of 30-day rolling MAE → alert
- If bias vector shifts by > 2° cosine distance in 1 week → flag for retraining

### 3.3 Dashboard
- Grafana panel: H10 MAE vs WT+DOW MAE (24h rolling)
- Grafana panel: Bias vector heatmap (24 hours × 7 days)
- Grafana panel: Engine selection split (WT+DOW vs DLinear+H10)

---

## 4. Retraining Strategy

### DLinear (base model)
- Retrain every 6 months on first day of January and July
- Process: train 6 folds from scratch (~30 min total on CPU)
- Trigger: manual via `python -m experiments.eval_h10 --config config.yaml`
- Checkpoints stored with version tag: `dlinear_v2_fold_1.pt`

### H10 Corrector (no retraining needed)
- Bias is online and continuously updated
- No retraining — it adapts automatically
- Exception: if bias diverges significantly, reset and cold-start with WT+DOW

### Versioning
- Model version stored in `models/dlinear/VERSION`
- Each SCADA actual stores which model version produced the prediction
- Allows post-hoc re-evaluation when new model versions are deployed

---

## 5. Failure Mode Analysis & Mitigations

| Failure Mode | Scenario | Mitigation | Detection |
|-------------|----------|------------|-----------|
| **Cold start** | First 2-7 days after deployment have zero bias | Pre-populate with WT+DOW error history | Bias magnitude < 10 MW → still warming |
| **Regime shift** | Sudden demand change (new factory, COVID-like event) | Automatic: H10 adapts within 1-3 days | Bias > 3σ historical std |
| **Holiday** | Christmas demand drops 30% | H10 corrects the day after; holiday calendar overlay planned | Calendar match → reduce bias weight |
| **Error cascade** | One catastrophic DLinear failure → over-correction next day | Clamp bias to ±2× historical std per hour | Per-hour bias > clamp threshold |
| **Data gap** | SCADA missing for >4h | Freeze bias at last known value | SCADA staleness alert |
| **Bias saturation** | Bias accumulates in one direction over weeks | Periodic reset to 7-day rolling window | Bias trend > 5 MW/week |
| **Ensemble failure** | All 6 DLinear folds disagree | Use median instead of mean, alert on std > 50 MW | Fold std > 50 MW |

---

## 6. Code Implementation Plan

### Files to create:
```
Backend/app/ml/dlinear_engine.py      — DLinear inference, fold ensemble, H10 correction
Backend/app/ml/h10_corrector.py        — H10 stateful corrector with persistence
Backend/tests/test_dlinear_engine.py   — Unit tests
Backend/tests/test_h10_corrector.py    — H10 unit tests
```

### Files to modify:
```
Backend/app/ml/__init__.py             — Export new classes
Backend/app/ml/weighted_trend_engine.py — No changes (stays as fallback)
Backend/app/services/baseline_forecast_service.py — Add engine routing
Backend/models/                         — Add dlinear/ directory with checkpoints
```

### Implementation steps:

1. **Copy DLinear code** from `DL_RESEARCH/experiments/eval_h10.py` (lines 20–46) into `dlinear_engine.py`
2. **Add fold ensemble**: load all 6 checkpoints, average predictions
3. **Add H10 corrector** with Redis/SQLite persistence
4. **Add engine selector** to `baseline_forecast_service.py`
5. **Add API parameter** `?engine=dlinear|wtdow`
6. **Add monitoring metrics** (Prometheus counters/gauges)
7. **Write tests**: unit tests for corrector logic, integration tests for engine selector
8. **Deploy shadow mode**, then canary, then full rollout

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| H10 over-corrects on holiday | Medium | Medium | Holiday calendar overlay (Phase 4) |
| DLinear checkpoint format changes | Low | High | Lock checkpoint format in CI test |
| SCADA delay causes stale bias | Medium | Low | Freeze bias, alert operator |
| Memory leak in error_buffer | Low | Low | Fixed-size deque (maxlen=48h/24=2) |
| Python version mismatch | Low | Medium | Pin Python 3.10 in deployment |
| Latency spike from 6-fold ensemble | Low | Low | Pre-warm model, use ONNX if needed |

---

## 8. Production Go/No-Go Criteria

### Go criteria:
- [ ] Shadow mode: H10 MAE ≤ WT+DOW MAE for 7 consecutive days
- [ ] Canary: No NaN/Inf predictions in 2 weeks
- [ ] Canary: p99 latency < 2s
- [ ] Cascade test: recovery within 1 day after fault injection
- [ ] Regime shift test: adaptation within 3 days
- [ ] All unit tests pass
- [ ] Bias serialization/deserialization tested with restart

### No-Go criteria:
- [ ] H10 MAE > 98 MW (WT+DOW baseline) in shadow mode
- [ ] Any NaN prediction in canary traffic
- [ ] p99 latency > 5s
- [ ] Unable to serialize/restore bias state
