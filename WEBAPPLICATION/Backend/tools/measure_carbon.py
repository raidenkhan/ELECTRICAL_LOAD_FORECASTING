"""Measure carbon footprint of training one DLinear fold using codecarbon.
Saves model separately — does NOT touch production checkpoints."""
import json, sys, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from codecarbon import EmissionsTracker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.ml.dlinear_engine import _DLinear, FEATURE_COLS

PROJECT = Path(__file__).resolve().parent.parent.parent.parent
CSV_PATH = PROJECT / "data" / "ecg_actual_demand_clean_with_temp.csv"
CKPT_DIR = Path(__file__).resolve().parent.parent / "models" / "dlinear" / "carbon_test"
STATS_PATH = CKPT_DIR / "normalization_stats.json"

INPUT_WINDOW = 168
FORECAST_HORIZON = 24
BATCH_SIZE = 4096
MAX_EPOCHS = 200
PATIENCE = 15
LR = 0.001
N_FEATURES = 8

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

# Use the largest fold (Fold_6: 2018-2025 train, 2026 test)
fold_name = "Fold_6"
train_start, train_end = "2018-01-01", "2025-12-31"
test_start, test_end = "2026-01-01", "2026-05-01"

print("=" * 60)
print("  DLinear Carbon Footprint Measurement")
print(f"  Fold: {fold_name} ({train_start} -> {train_end})")
print("=" * 60)

CKPT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading data...")
df = load_and_prepare()
print(f"  {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

train_mask = (df['date'] >= pd.Timestamp(train_start)) & (df['date'] <= pd.Timestamp(train_end))
test_mask = (df['date'] >= pd.Timestamp(test_start)) & (df['date'] <= pd.Timestamp(test_end))
df_tr = df[train_mask].copy()
df_te = df[test_mask].copy()
print(f"  Train: {len(df_tr)} rows, Test: {len(df_te)} rows")

df_tr_n, means, stds = normalize(df_tr, FEATURE_COLS)
df_te_n, _, _ = normalize(df_te, FEATURE_COLS, means, stds)

# Save stats
with open(STATS_PATH, "w") as f:
    json.dump({fold_name: {"means": {k: float(v) for k, v in means.items()},
                            "stds": {k: float(v) for k, v in stds.items()}}}, f, indent=2)

X_tr, y_tr = to_sequences(df_tr_n, INPUT_WINDOW, FORECAST_HORIZON)
X_te, y_te = to_sequences(df_te_n, INPUT_WINDOW, FORECAST_HORIZON)
print(f"  Sequences: train {len(X_tr)}, test {len(X_te)}")

model = _DLinear(N_FEATURES, FORECAST_HORIZON, INPUT_WINDOW).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss()

tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr).float(), torch.tensor(y_tr).float()),
                       batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
te_loader = DataLoader(TensorDataset(torch.tensor(X_te).float(), torch.tensor(y_te).float()),
                       batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {n_params:,}")

print("\nStarting training with CodeCarbon tracking...")
tracker = EmissionsTracker(
    output_dir=str(CKPT_DIR),
    output_file="emissions.csv",
    project_name="dlinear_fold6_train",
    log_level="warning",
)
tracker.start()

t0 = time.time()
best, best_state, stale = float("inf"), None, 0
n_epochs_run = 0

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
    n_epochs_run = epoch
    if val_loss < best:
        best, best_state, stale = val_loss, {k: v.cpu() for k, v in model.state_dict().items()}, 0
    else:
        stale += 1
        if stale >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (val L1={best:.4f})")
            break
    if epoch == 1 or epoch % 20 == 0:
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}: val L1={val_loss:.4f} (best={best:.4f}) [{elapsed:.0f}s]")

elapsed = time.time() - t0

emissions = tracker.stop()
print(f"\nTraining complete: {n_epochs_run} epochs in {elapsed:.0f}s")

# Save checkpoint
ckpt_path = CKPT_DIR / f"h10_{fold_name}.pt"
torch.save(best_state, ckpt_path)
print(f"  Saved: {ckpt_path}")
print(f"  Size: {os.path.getsize(ckpt_path) / 1024:.1f} KB")

# Test evaluation
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
model.eval()
te_loader = DataLoader(TensorDataset(torch.tensor(X_te).float(), torch.tensor(y_te).float()),
                       batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
with torch.no_grad():
    preds = torch.cat([model(x.to(device)).cpu() for x, _ in te_loader]).numpy()
mae_val = float(np.mean(np.abs(preds - y_te)))
denorm_mae = mae_val * stds['demand_mw']
print(f"  Test normalized MAE: {mae_val:.4f} -> {denorm_mae:.1f} MW")

print("\n" + "=" * 60)
print("  CARBON FOOTPRINT SUMMARY")
print("=" * 60)
print(f"  Model: DLinear ({n_params:,} parameters)")
print(f"  Hardware: {device}")
print(f"  Training data: {len(df_tr)} rows ({train_start} to {train_end})")
print(f"  Epochs run: {n_epochs_run} (patience={PATIENCE})")
print(f"  Wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
if isinstance(emissions, (int, float)):
    print(f"  CO₂ emissions: {emissions:.6f} kg")
elif isinstance(emissions, dict):
    print(f"  CO₂ emissions: {emissions.get('co2_kg', 'N/A')} kg")
    print(f"  Energy: {emissions.get('energy_kwh', 'N/A')} kWh")
else:
    print(f"  CO₂ emissions: {emissions}")
print(f"  Full report: {CKPT_DIR / 'emissions.csv'}")
print("=" * 60)
