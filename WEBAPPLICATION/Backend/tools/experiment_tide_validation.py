"""TIDE validation: bootstrap CIs, comparison baselines, alpha sensitivity.

Loads each DLinear fold checkpoint, runs inference with:
- No correction (baseline)
- TIDE (alpha=0.3)
- Simple Moving Average (N=7, 14, 30)
- Kalman filter (Q/R sweep)
- Linear error trend extrapolation

Then bootstraps CIs on all results and sweeps alpha for TIDE.

Usage: py -3.10 tools/experiment_tide_validation.py
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.ml.dlinear_engine import _DLinear, FEATURE_COLS

INPUT_WINDOW = 168
FORECAST_HORIZON = 24
BATCH_SIZE = 4096
N_FEATURES = 8
N_BOOTSTRAP = 2000
ALPHA_CI = 0.95

FOLDS = [
    ("Fold_1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("Fold_2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("Fold_3", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("Fold_4", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("Fold_5", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("Fold_6", "2018-01-01", "2025-12-31", "2026-01-01", "2026-05-01"),
]

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE_DIR / "models" / "dlinear"
OUT_DIR = BASE_DIR / "models" / "dlinear" / "tide_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def load_and_prepare():
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    csv_path = project_root / "data" / "ecg_actual_demand_clean_with_temp.csv"
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'hour']).reset_index(drop=True)
    df['demand_mw'] = df['demand_mw'].astype(float)
    df['temperature_c'] = df['temp_c'].astype(float) if 'temp_c' in df.columns else 28.0
    ts = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'] - 1, unit='h')
    df['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * ts.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * ts.dt.month / 12)
    return df

def normalize(df, feature_cols, means=None, stds=None):
    result = df.copy()
    if means is None:
        means, stds = {}, {}
        for c in feature_cols:
            v = df[c].values.astype(np.float32)
            means[c] = float(np.nanmean(v))
            stds[c] = float(np.nanstd(v)) + 1e-8
            result[c] = (v - means[c]) / stds[c]
    else:
        for c in feature_cols:
            result[c] = (df[c].values.astype(np.float32) - means[c]) / stds[c]
    return result, means, stds

def to_sequences(df, iw, fh):
    cols = FEATURE_COLS
    features = df[cols].values.astype(np.float32)
    target = df['demand_mw'].values.astype(np.float32)
    n = len(df) - iw - fh + 1
    if n < 1:
        return np.array([]), np.array([])
    X, y = [], []
    for i in range(n):
        X.append(features[i: i + iw])
        y.append(target[i + iw: i + iw + fh])
    return np.array(X), np.array(y)

# ---------------------------------------------------------------------------
# Correctors
# ---------------------------------------------------------------------------
def correct_tide(preds, actuals, alpha=0.3, warmup=48):
    """TIDE: EMA of recent errors. preds/actuals: (n_samples, 24)."""
    n = len(preds)
    corrected = preds.copy()
    bias_ema = 0.0
    for i in range(n):
        corrected[i] = preds[i] - bias_ema
        if i >= warmup:
            err = np.mean(preds[i] - actuals[i])
            if i == warmup:
                bias_ema = err
            else:
                bias_ema = alpha * err + (1 - alpha) * bias_ema
    return corrected

def correct_sma(preds, actuals, n_days=7):
    """Simple Moving Average of errors over last N days."""
    window = n_days
    n = len(preds)
    corrected = preds.copy()
    err_buffer = []
    for i in range(n):
        if i > 0:
            err = np.mean(preds[i-1] - actuals[i-1])
            err_buffer.append(err)
            if len(err_buffer) > window:
                err_buffer.pop(0)
        if len(err_buffer) >= 1:
            corrected[i] = preds[i] - np.mean(err_buffer)
    return corrected

def correct_kalman(preds, actuals, q=1e-4, r=1.0):
    """Kalman filter: state = bias, observation = error."""
    n = len(preds)
    corrected = preds.copy()
    bias = 0.0
    p = 1.0
    for i in range(n):
        corrected[i] = preds[i] - bias
        if i > 0:
            err = np.mean(preds[i-1] - actuals[i-1])
            k = p / (p + r)
            bias = bias + k * (err - bias)
            p = (1 - k) * p + q
    return corrected

def correct_linear_trend(preds, actuals, window=14):
    """Linear trend extrapolation of recent errors."""
    n = len(preds)
    corrected = preds.copy()
    err_history = []
    for i in range(n):
        if len(err_history) >= 2:
            x = np.arange(len(err_history))
            coeffs = np.polyfit(x, err_history, 1)
            pred_err = coeffs[0] * len(err_history) + coeffs[1]
            corrected[i] = preds[i] - pred_err
        if i > 0:
            err_history.append(np.mean(preds[i-1] - actuals[i-1]))
            if len(err_history) > window:
                err_history.pop(0)
    return corrected

# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------
def bootstrap_ci(mae_values, n_iter=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap CI for mean of MAE values across folds."""
    n = len(mae_values)
    means = np.zeros(n_iter)
    rng = np.random.default_rng(42)
    for i in range(n_iter):
        sample = rng.choice(mae_values, size=n, replace=True)
        means[i] = np.mean(sample)
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(np.mean(mae_values)), float(lo), float(hi)

def paired_bootstrap_test(baseline_errors, corrected_errors, n_iter=N_BOOTSTRAP):
    """Paired bootstrap test: fraction of iterations where corrected > baseline."""
    n = len(baseline_errors)
    rng = np.random.default_rng(43)
    count_worse = 0
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        if np.mean(corrected_errors[idx]) > np.mean(baseline_errors[idx]):
            count_worse += 1
    return count_worse / n_iter

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  TIDE Validation: CIs, Baselines, Alpha Sensitivity")
    print("=" * 60)

    print("\nLoading data...")
    df = load_and_prepare()
    print(f"Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    all_results = []

    for name, tr_s, tr_e, te_s, te_e in FOLDS:
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        ckpt_path = CKPT_DIR / f"h10_{name}.pt"
        if not ckpt_path.exists():
            print(f"  SKIP: checkpoint not found: {ckpt_path}")
            continue

        train_mask = (df['date'] >= pd.Timestamp(tr_s)) & (df['date'] <= pd.Timestamp(tr_e))
        test_mask = (df['date'] >= pd.Timestamp(te_s)) & (df['date'] <= pd.Timestamp(te_e))
        df_tr = df[train_mask].copy()
        df_te = df[test_mask].copy()

        if len(df_te) < INPUT_WINDOW + FORECAST_HORIZON:
            print(f"  SKIP: test too small ({len(df_te)} rows)")
            continue

        df_tr_n, means, stds = normalize(df_tr, FEATURE_COLS)
        df_te_n, _, _ = normalize(df_te, FEATURE_COLS, means, stds)

        X_te, y_te = to_sequences(df_te_n, INPUT_WINDOW, FORECAST_HORIZON)
        print(f"  Test sequences: {len(X_te)}")

        # Load model
        model = _DLinear(N_FEATURES, FORECAST_HORIZON, INPUT_WINDOW).to(device)
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get('model_state_dict', state)
        model.load_state_dict(sd)
        model.eval()

        # Generate predictions
        te_loader = DataLoader(TensorDataset(torch.tensor(X_te).float(), torch.tensor(y_te).float()),
                               batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
        with torch.no_grad():
            raw_preds = torch.cat([model(x.to(device)).cpu() for x, _ in te_loader]).numpy()

        # Denormalize predictions back to MW space
        demand_std = stds['demand_mw']
        demand_mean = means['demand_mw']
        raw_preds_mw = raw_preds * demand_std + demand_mean
        y_te_mw = y_te * demand_std + demand_mean

        # ---- Evaluate all correctors ----
        configs = {
            "baseline": lambda p, a: p,
            "tide_a0.1": lambda p, a: correct_tide(p, a, alpha=0.1),
            "tide_a0.3": lambda p, a: correct_tide(p, a, alpha=0.3),
            "tide_a0.5": lambda p, a: correct_tide(p, a, alpha=0.5),
            "tide_a0.7": lambda p, a: correct_tide(p, a, alpha=0.7),
            "tide_a0.9": lambda p, a: correct_tide(p, a, alpha=0.9),
            "sma_7d": lambda p, a: correct_sma(p, a, n_days=7),
            "sma_14d": lambda p, a: correct_sma(p, a, n_days=14),
            "sma_30d": lambda p, a: correct_sma(p, a, n_days=30),
            "kalman_q1e4_r1": lambda p, a: correct_kalman(p, a, q=1e-4, r=1.0),
            "kalman_q1e3_r1": lambda p, a: correct_kalman(p, a, q=1e-3, r=1.0),
            "kalman_q1e2_r1": lambda p, a: correct_kalman(p, a, q=1e-2, r=1.0),
            "linear_trend_14d": lambda p, a: correct_linear_trend(p, a, window=14),
        }

        fold_mae = {}
        for cfg_name, corrector_fn in configs.items():
            corrected = corrector_fn(raw_preds_mw, y_te_mw)
            mae = float(np.mean(np.abs(corrected - y_te_mw)))
            bias = float(np.mean(corrected - y_te_mw))
            fold_mae[cfg_name] = {"mae": mae, "bias": bias}
            print(f"  {cfg_name:20s}: MAE={mae:.1f} MW, bias={bias:+.1f} MW")

        # Bootstrap CIs for this fold
        for cfg_name, v in fold_mae.items():
            corrected = corrector_fn(raw_preds_mw, y_te_mw)
            hourly_errors = np.abs(corrected - y_te_mw).ravel()
            mean_mae, ci_lo, ci_hi = bootstrap_ci(hourly_errors, n_iter=N_BOOTSTRAP)
            all_results.append({
                "fold": name,
                "corrector": cfg_name,
                "mae": v["mae"],
                "bias": v["bias"],
                "ci_lo": ci_lo,
                "ci_hi": ci_hi
            })

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY WITH BOOTSTRAP 95% CIS")
    print(f"{'='*60}")

    df_res = pd.DataFrame(all_results)

    # Per-corrector mean across folds
    for corrector in df_res['corrector'].unique():
        sub = df_res[df_res['corrector'] == corrector]
        mean_mae = sub['mae'].mean()
        mean_bias = sub['bias'].mean()
        print(f"\n  {corrector:20s}:")
        print(f"    Mean MAE:  {mean_mae:.1f} MW (bias={mean_bias:+.1f} MW)")
        for _, r in sub.iterrows():
            print(f"      {r['fold']}: MAE={r['mae']:.1f} [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]")

    # Paired significance: TIDE vs each baseline
    print(f"\n{'='*60}")
    print("  PAIRED SIGNIFICANCE TESTS (vs baseline)")
    print(f"{'='*60}")
    for cfg_name in configs:
        if cfg_name == "baseline":
            continue
        sub = df_res[df_res['corrector'] == cfg_name]
        base = df_res[df_res['corrector'] == "baseline"]
        merged = sub.merge(base, on='fold', suffixes=('_corr', '_base'))
        deltas = merged['mae_corr'].values - merged['mae_base'].values
        better = np.sum(deltas < 0)
        p = paired_bootstrap_test(merged['mae_base'].values, merged['mae_corr'].values)
        print(f"  {cfg_name:20s}: {better}/{len(deltas)} folds better, p={p:.4f}")

    # Save results
    df_res.to_csv(OUT_DIR / "results.csv", index=False)
    print(f"\nResults saved to {OUT_DIR / 'results.csv'}")

    # Generate summary JSON
    summary = {}
    for corrector in df_res['corrector'].unique():
        sub = df_res[df_res['corrector'] == corrector]
        mean_mae = sub['mae'].mean()
        mean_bias = sub['bias'].mean()
        ci_lo = sub['ci_lo'].values
        ci_hi = sub['ci_hi'].values
        summary[corrector] = {
            "mean_mae": float(mean_mae),
            "mean_bias": float(mean_bias),
            "ci_lo_mean": float(np.mean(ci_lo)),
            "ci_hi_mean": float(np.mean(ci_hi)),
            "fold_results": {
                r['fold']: {"mae": r['mae'], "bias": r['bias'], "ci_lo": r['ci_lo'], "ci_hi": r['ci_hi']}
                for _, r in sub.iterrows()
            }
        }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {OUT_DIR / 'summary.json'}")
