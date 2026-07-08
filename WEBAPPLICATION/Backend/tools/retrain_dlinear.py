"""Retrain 6-fold DLinear ensemble on full dataset (2018-2026)."""
import json, sys, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.ml.dlinear_engine import _DLinear, FEATURE_COLS

PROJECT = Path(__file__).resolve().parent.parent.parent.parent  # LOADFORECASINGPROJECT
CSV_PATH = PROJECT / "data" / "ecg_actual_demand_clean_with_temp.csv"
CKPT_DIR = Path(__file__).resolve().parent.parent / "models" / "dlinear"
STATS_PATH = CKPT_DIR / "normalization_stats.json"

INPUT_WINDOW = 168
FORECAST_HORIZON = 24
BATCH_SIZE = 4096
MAX_EPOCHS = 200
PATIENCE = 15
LR = 0.001
N_FEATURES = 8

FOLDS = [
    ("Fold_1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("Fold_2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("Fold_3", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("Fold_4", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("Fold_5", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("Fold_6", "2018-01-01", "2025-12-31", "2026-01-01", "2026-05-01"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_and_prepare():
    df = pd.read_csv(CSV_PATH)
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


def train():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from {CSV_PATH}")
    df = load_and_prepare()
    print(f"Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Demand: mean={df['demand_mw'].mean():.0f}, std={df['demand_mw'].std():.0f}")

    all_stats = {}
    for name, tr_s, tr_e, te_s, te_e in FOLDS:
        ckpt_path = CKPT_DIR / f"h10_{name}.pt"
        train_mask = (df['date'] >= pd.Timestamp(tr_s)) & (df['date'] <= pd.Timestamp(tr_e))
        test_mask = (df['date'] >= pd.Timestamp(te_s)) & (df['date'] <= pd.Timestamp(te_e))
        df_tr = df[train_mask].copy()
        df_te = df[test_mask].copy()
        print(f"\n{'='*60}")
        print(f"  {name}: train {tr_s} -> {tr_e} ({len(df_tr)} rows), test {te_s} -> {te_e} ({len(df_te)} rows)")
        print(f"{'='*60}")

        if len(df_te) < INPUT_WINDOW + FORECAST_HORIZON:
            print(f"  SKIP: test set too small ({len(df_te)} rows, need {INPUT_WINDOW + FORECAST_HORIZON})")
            continue

        df_tr_n, means, stds = normalize(df_tr, FEATURE_COLS)
        df_te_n, _, _ = normalize(df_te, FEATURE_COLS, means, stds)
        all_stats[name] = {"means": means, "stds": stds}

        X_tr, y_tr = to_sequences(df_tr_n, INPUT_WINDOW, FORECAST_HORIZON)
        X_te, y_te = to_sequences(df_te_n, INPUT_WINDOW, FORECAST_HORIZON)
        print(f"  Sequences: train {len(X_tr)}, test {len(X_te)}")

        model = _DLinear(N_FEATURES, FORECAST_HORIZON, INPUT_WINDOW).to(device)

        if ckpt_path.exists():
            print(f"  Loading existing checkpoint: {ckpt_path.name}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            criterion = nn.L1Loss()
            tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr).float(), torch.tensor(y_tr).float()),
                                   batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            te_loader = DataLoader(TensorDataset(torch.tensor(X_te).float(), torch.tensor(y_te).float()),
                                   batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

            best, best_state, stale = float("inf"), None, 0
            t0 = time.time()
            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                for x, y in tr_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    criterion(model(x), y).backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    preds = torch.cat([model(x.to(device)).cpu() for x, _ in te_loader])
                    val_loss = float(criterion(preds, torch.tensor(y_te)).item())
                if val_loss < best:
                    best, best_state, stale = val_loss, {k: v.cpu() for k, v in model.state_dict().items()}, 0
                else:
                    stale += 1
                    if stale >= PATIENCE:
                        print(f"  Early stop at epoch {epoch} (val L1={best:.4f})")
                        break
                if epoch == 1 or epoch % 20 == 0:
                    print(f"  Epoch {epoch:3d}: val L1={val_loss:.4f} (best={best:.4f})")
            elapsed = time.time() - t0
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            torch.save(best_state, ckpt_path)
            print(f"  Saved {ckpt_path.name} ({elapsed:.0f}s, best val L1={best:.4f})")

        # Quick eval on test set
        model.eval()
        te_loader = DataLoader(TensorDataset(torch.tensor(X_te).float(), torch.tensor(y_te).float()),
                               batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
        with torch.no_grad():
            preds = torch.cat([model(x.to(device)).cpu() for x, _ in te_loader]).numpy()
            actuals = y_te
        mae_val = float(np.mean(np.abs(preds - actuals)))
        denorm_mae = mae_val * stds['demand_mw']
        print(f"  Test normalized MAE: {mae_val:.4f}   ->   {denorm_mae:.1f} MW")

    # Save normalization stats
    serializable = {}
    for name, stat in all_stats.items():
        serializable[name] = {
            "means": {k: float(v) for k, v in stat["means"].items()},
            "stds": {k: float(v) for k, v in stat["stds"].items()},
        }
    with open(STATS_PATH, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved normalization stats: {STATS_PATH}")
    print(f"Checkpoints in: {CKPT_DIR}")


if __name__ == "__main__":
    train()
