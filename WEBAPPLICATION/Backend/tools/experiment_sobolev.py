"""Sobolev Trajectory Loss ablation - DLinear 6-fold x lambda ? {0.0, 0.3, 1.0}.

Hypothesis: adding a first-difference (ramp rate) penalty to the loss function
improves forecast trajectory accuracy without degrading pointwise MAE.

lambda=0.0 (baseline) loads existing checkpoints.
lambda=0.3, 1.0 train from scratch with SobolevLoss.

Results written incrementally to models/dlinear/sobolev_experiment/results.csv
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.ml.dlinear_engine import _DLinear, FEATURE_COLS

LAMBDA_VALUES = [0.0, 0.3, 1.0]
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

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE_DIR / "models" / "dlinear"
RESULTS_DIR = BASE_DIR / "models" / "dlinear" / "sobolev_experiment"
RESULTS_CSV = RESULTS_DIR / "results.csv"
SUMMARY_FILE = RESULTS_DIR / "summary.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Results dir: {RESULTS_DIR}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sobolev Loss
# ---------------------------------------------------------------------------
class SobolevLoss(nn.Module):
    """L1 + lambda ? L1 of first differences (ramp-rate penalty)."""
    def __init__(self, lambd: float = 0.3):
        super().__init__()
        self.lambd = lambd
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pointwise = self.l1(pred, target)
        if self.lambd == 0.0 or pred.shape[-1] < 2:
            return pointwise
        dp = pred[..., 1:] - pred[..., :-1]
        dt = target[..., 1:] - target[..., :-1]
        ramp = self.l1(dp, dt)
        return pointwise + self.lambd * ramp


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(pred - target)))
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    ramp_mae = float(np.mean(np.abs(dp - dt)))
    peak_ramp = float(np.max(np.abs(dp - dt)))
    return {"mae": mae, "ramp_mae": ramp_mae, "peak_ramp": peak_ramp}


# ---------------------------------------------------------------------------
# Data pipeline (mirrors retrain_dlinear.py)
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
# Training
# ---------------------------------------------------------------------------
def train_fold(name, tr_s, tr_e, te_s, te_e, lambd):
    ckpt_path = CKPT_DIR / f"h10_{name}.pt"
    seed = hash((name, lambd)) % (2**31)

    train_mask = (df['date'] >= pd.Timestamp(tr_s)) & (df['date'] <= pd.Timestamp(tr_e))
    test_mask = (df['date'] >= pd.Timestamp(te_s)) & (df['date'] <= pd.Timestamp(te_e))
    df_tr = df[train_mask].copy()
    df_te = df[test_mask].copy()

    if len(df_te) < INPUT_WINDOW + FORECAST_HORIZON:
        print(f"  SKIP: test too small ({len(df_te)} rows)")
        return None

    df_tr_n, means, stds = normalize(df_tr, FEATURE_COLS)
    df_te_n, _, _ = normalize(df_te, FEATURE_COLS, means, stds)

    X_tr, y_tr = to_sequences(df_tr_n, INPUT_WINDOW, FORECAST_HORIZON)
    X_te, y_te = to_sequences(df_te_n, INPUT_WINDOW, FORECAST_HORIZON)
    print(f"  Sequences: train {len(X_tr)}, test {len(X_te)}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _DLinear(N_FEATURES, FORECAST_HORIZON, INPUT_WINDOW).to(device)

    # For lambda=0.0, load existing checkpoint
    if lambd == 0.0 and ckpt_path.exists():
        print(f"  Loading baseline checkpoint: {ckpt_path.name}")
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get('model_state_dict', state)
        model.load_state_dict(sd)
        model.eval()
        # Evaluate
        te_loader = DataLoader(TensorDataset(torch.tensor(X_te).float(), torch.tensor(y_te).float()),
                               batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
        with torch.no_grad():
            preds = torch.cat([model(x.to(device)).cpu() for x, _ in te_loader]).numpy()
        metrics = compute_metrics(preds, y_te)
        metrics["epochs_trained"] = 0
        metrics["elapsed_sec"] = 0
        return metrics

    # Train from scratch
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = SobolevLoss(lambd=lambd)
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
                if epoch == 1 or epoch % 10 == 0 or epoch < 20:
                    print(f"  Early stop at epoch {epoch} (val={best:.4f})")
                break
        if epoch == 1 or epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: val={val_loss:.4f} (best={best:.4f})")

    elapsed = time.time() - t0
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(x.to(device)).cpu() for x, _ in te_loader]).numpy()
    metrics = compute_metrics(preds, y_te)
    metrics["epochs_trained"] = next((e for e in range(1, MAX_EPOCHS + 1) if e == 1 or e % 20 == 0), MAX_EPOCHS)
    metrics["elapsed_sec"] = round(elapsed, 1)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Sobolev Trajectory Loss Experiment")
    print("  Lambda values:", LAMBDA_VALUES)
    print("  Folds:", [f[0] for f in FOLDS])
    print("=" * 60)

    global df
    print("\nLoading data...")
    df = load_and_prepare()
    print(f"Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Demand: mean={df['demand_mw'].mean():.0f}, std={df['demand_mw'].std():.0f}")

    all_results = []
    header = ["fold", "lambda", "mae", "ramp_mae", "peak_ramp", "epochs", "elapsed_sec"]
    if not RESULTS_CSV.exists():
        pd.DataFrame(columns=header).to_csv(RESULTS_CSV, index=False)

    for name, tr_s, tr_e, te_s, te_e in FOLDS:
        for lambd in LAMBDA_VALUES:
            print(f"\n{'-' * 50}")
            print(f"  {name} | lambda = {lambd}")
            print(f"{'-' * 50}")
            t0 = time.time()
            metrics = train_fold(name, tr_s, tr_e, te_s, te_e, lambd)
            wall = time.time() - t0
            if metrics is None:
                print(f"  SKIPPED")
                continue

            row = {
                "fold": name,
                "lambda": lambd,
                "mae": round(metrics["mae"], 6),
                "ramp_mae": round(metrics["ramp_mae"], 6),
                "peak_ramp": round(metrics["peak_ramp"], 6),
                "epochs": metrics.get("epochs_trained", 0),
                "elapsed_sec": round(metrics.get("elapsed_sec", wall), 1),
            }
            all_results.append(row)

            # Append incremental row to CSV
            pd.DataFrame([row]).to_csv(RESULTS_CSV, mode='a', header=False, index=False)

            # Print summary
            denorm_std = df['demand_mw'].std()
            print(f"  -> MAE={metrics['mae']:.4f} ({metrics['mae']*denorm_std:.1f} MW)  "
                  f"ramp_MAE={metrics['ramp_mae']:.4f}  "
                  f"peak_ramp={metrics['peak_ramp']:.4f}  [{wall:.0f}s]")

    # Final summary
    df_res = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    denorm_std = df['demand_mw'].std()
    for lambd in LAMBDA_VALUES:
        sub = df_res[df_res["lambda"] == lambd]
        print(f"\n  lambda = {lambd}")
        print(f"    Mean MAE:       {sub['mae'].mean():.4f} ({sub['mae'].mean()*denorm_std:.1f} MW)")
        print(f"    Mean ramp_MAE:  {sub['ramp_mae'].mean():.4f}")
        print(f"    Mean peak_ramp: {sub['peak_ramp'].mean():.4f}")
        for _, r in sub.iterrows():
            print(f"      {r['fold']}: MAE={r['mae']:.4f}  ramp={r['ramp_mae']:.4f}  peak={r['peak_ramp']:.4f}")

    # Save summary JSON
    summary = {}
    for lambd in LAMBDA_VALUES:
        sub = df_res[df_res["lambda"] == lambd]
        summary[f"lambda_{lambd}"] = {
            "mean_mae": float(sub["mae"].mean()),
            "mean_mae_mw": float(sub["mae"].mean() * denorm_std),
            "mean_ramp_mae": float(sub["ramp_mae"].mean()),
            "mean_peak_ramp": float(sub["peak_ramp"].mean()),
            "fold_results": {r["fold"]: {"mae": r["mae"], "ramp_mae": r["ramp_mae"], "peak_ramp": r["peak_ramp"]}
                             for _, r in sub.iterrows()}
        }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_CSV}")
    print(f"Summary saved to {SUMMARY_FILE}")
