"""Compare 2yr / 4yr / full-history DLinear training on 2026-H1 test set.
Tests whether training on just recent years equals the full 6-fold ensemble.
"""
import sys, os, json, time, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.ml.dlinear_engine import _DLinear, FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # LOADFORECASINGPROJECT
CSV = ROOT / "data" / "ecg_actual_demand_clean_with_temp.csv"
CKPT_DIR = Path(__file__).resolve().parent.parent / "models" / "dlinear"
OUT = CKPT_DIR / "window_size_experiment"
OUT.mkdir(exist_ok=True)

IW, FH, BS, LR = 168, 24, 4096, 0.001
MAX_EPOCHS, PATIENCE = 100, 10

device = torch.device("cpu")

def load_data():
    df = pd.read_csv(CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date','hour']).reset_index(drop=True)
    df['demand_mw'] = df['demand_mw'].astype(float)
    df['temperature_c'] = df['temp_c'].astype(float)
    ts = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour']-1, unit='h')
    df['hour_sin'] = np.sin(2*np.pi*ts.dt.hour/24)
    df['hour_cos'] = np.cos(2*np.pi*ts.dt.hour/24)
    df['dow_sin'] = np.sin(2*np.pi*ts.dt.dayofweek/7)
    df['dow_cos'] = np.cos(2*np.pi*ts.dt.dayofweek/7)
    df['month_sin'] = np.sin(2*np.pi*ts.dt.month/12)
    df['month_cos'] = np.cos(2*np.pi*ts.dt.month/12)
    return df

def normalize(df, means=None, stds=None):
    r = df.copy()
    if means is None:
        means, stds = {}, {}
        for c in FEATURE_COLS:
            v = df[c].values.astype(np.float32)
            means[c] = float(np.nanmean(v))
            stds[c] = float(np.nanstd(v)) + 1e-8
            r[c] = (v - means[c]) / stds[c]
    else:
        for c in FEATURE_COLS:
            r[c] = (df[c].values.astype(np.float32) - means[c]) / stds[c]
    return r, means, stds

def to_seq(df):
    cols = FEATURE_COLS
    f = df[cols].values.astype(np.float32)
    t = df['demand_mw'].values.astype(np.float32)
    n = len(df) - IW - FH + 1
    if n < 1: return np.array([]), np.array([])
    X, y = [], []
    for i in range(n):
        X.append(f[i:i+IW])
        y.append(t[i+IW:i+IW+FH])
    return np.array(X), np.array(y)

def train_model(name, train_start, train_end, test_start, test_end, df):
    print(f"\n{'='*60}")
    print(f"  {name}: train {train_start}->{train_end}, test {test_start}->{test_end}")
    print(f"{'='*60}")
    train_mask = (df['date'] >= pd.Timestamp(train_start)) & (df['date'] <= pd.Timestamp(train_end))
    test_mask = (df['date'] >= pd.Timestamp(test_start)) & (df['date'] <= pd.Timestamp(test_end))
    df_tr = df[train_mask].copy()
    df_te = df[test_mask].copy()
    
    df_tr_n, means, stds = normalize(df_tr)
    df_te_n, _, _ = normalize(df_te, means, stds)
    
    X_tr, y_tr = to_seq(df_tr_n)
    X_te, y_te = to_seq(df_te_n)
    print(f"  Sequences: train {len(X_tr)}, test {len(X_te)}")
    
    model = _DLinear(len(FEATURE_COLS), FH, IW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()
    
    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr).float(), torch.tensor(y_tr).float()),
                           batch_size=BS, shuffle=True)
    te_loader = DataLoader(TensorDataset(torch.tensor(X_te).float(), torch.tensor(y_te).float()),
                           batch_size=BS*2, shuffle=False)
    
    best, best_state, stale = float('inf'), None, 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        for x, y in tr_loader:
            opt.zero_grad()
            criterion(model(x), y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = torch.cat([model(x) for x, _ in te_loader])
            val = float(criterion(preds, torch.tensor(y_te)).item())
        if val < best:
            best, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"  Early stop epoch {epoch} (val L1={best:.4f})")
                break
        if epoch==1 or epoch%20==0:
            print(f"  Epoch {epoch:3d}: val L1={val:.4f} (best={best:.4f})")
    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    
    # Final evaluation
    with torch.no_grad():
        preds_n = torch.cat([model(x) for x, _ in te_loader]).numpy()
    mae_n = float(np.mean(np.abs(preds_n - y_te)))
    mae_mw = mae_n * stds['demand_mw']
    print(f"  {name}: MAE={mae_n:.4f} (norm) = {mae_mw:.1f} MW ({elapsed:.0f}s)")
    
    # Save checkpoint
    ckpt = OUT / f"{name}.pt"
    torch.save(best_state, ckpt)
    print(f"  Saved {ckpt.name}")
    
    return {
        'name': name,
        'train': f'{train_start}->{train_end}',
        'test': f'{test_start}->{test_end}',
        'mae_norm': round(mae_n, 4),
        'mae_mw': round(mae_mw, 1),
        'train_rows': len(df_tr),
        'test_rows': len(df_te),
        'train_years': len(df_tr['date'].dt.year.unique()),
        'elapsed_s': round(elapsed),
    }

def evaluate_tide(model, df_te, means, stds, alpha=0.3):
    """Run TIDE on the model's predictions."""
    df_te_n, _, _ = normalize(df_te, means, stds)
    X_te, y_te = to_seq(df_te_n)
    model.eval()
    with torch.no_grad():
        preds_n = torch.cat([model(torch.tensor(X).float()) for X in torch.split(torch.tensor(X_te).float(), BS*2)]).numpy()
    
    # TIDE correction
    bias = 0.0
    corrected = []
    for i in range(len(preds_n)):
        corrected.append(preds_n[i] - bias)
        err = np.mean(preds_n[i] - y_te[i])
        bias = alpha * err + (1 - alpha) * bias
    
    corrected = np.array(corrected)
    mae_corr = float(np.mean(np.abs(corrected - y_te))) * stds['demand_mw']
    mae_raw = float(np.mean(np.abs(preds_n - y_te))) * stds['demand_mw']
    return mae_raw, mae_corr

def main():
    print(f"Loading data from {CSV}")
    df = load_data()
    print(f"Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Train models with different windows, all test on 2026-H1
    experiments = [
        ("2yr_2024_2025", "2024-01-01", "2025-12-31", "2026-01-01", "2026-05-01"),
        ("4yr_2022_2025", "2022-01-01", "2025-12-31", "2026-01-01", "2026-05-01"),
        ("8yr_2018_2025", "2018-01-01", "2025-12-31", "2026-01-01", "2026-05-01"),
    ]
    
    results = []
    for name, tr_s, tr_e, te_s, te_e in experiments:
        r = train_model(name, tr_s, tr_e, te_s, te_e, df)
        results.append(r)
    
    # Evaluate TIDE on each
    print(f"\n{'='*60}")
    print("  TIDE EVALUATION (alpha=0.3)")
    print(f"{'='*60}")
    for r in results:
        name = r['name']
        # Reload model
        model = _DLinear(len(FEATURE_COLS), FH, IW).to(device)
        model.load_state_dict(torch.load(OUT / f"{name}.pt", map_location=device))
        
        # Get test data
        test_mask = (df['date'] >= pd.Timestamp(r['test'].split('->')[0])) & \
                    (df['date'] <= pd.Timestamp(r['test'].split('->')[1]))
        df_te = df[test_mask].copy()
        
        # Normalize using training stats (we need means/stds - recompute from training)
        train_mask = (df['date'] >= pd.Timestamp(r['train'].split('->')[0])) & \
                     (df['date'] <= pd.Timestamp(r['train'].split('->')[1]))
        df_tr = df[train_mask].copy()
        _, means, stds = normalize(df_tr)
        
        mae_raw, mae_tide = evaluate_tide(model, df_te, means, stds)
        r['mae_raw_mw'] = round(mae_raw, 1)
        r['mae_tide_mw'] = round(mae_tide, 1)
        r['tide_improvement'] = round((mae_raw - mae_tide) / mae_raw * 100, 1)
        print(f"  {name:20s}: raw={mae_raw:.1f} MW -> TIDE={mae_tide:.1f} MW ({r['tide_improvement']:.1f}%)")
    
    # Compare with Fold_6 from tide_validation
    print(f"\n{'='*60}")
    print("  COMPARISON WITH Fold_6 (trained on 2018-2025)")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']:20s}: {r['train_years']}yr train, {r['train_rows']} rows, MAE={r['mae_mw']:.1f} MW, +TIDE={r['mae_tide_mw']:.1f} MW")
    
    # Save
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT / 'results.json'}")
    
    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Window':20s} {'Years':5s} {'Rows':8s} {'Raw MAE':10s} {'+TIDE':10s} {'Gain':8s}")
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for r in sorted(results, key=lambda x: x['mae_mw']):
        print(f"  {r['name']:20s} {r['train_years']:5d} {r['train_rows']:8d} {r['mae_mw']:8.1f}MW {r['mae_tide_mw']:8.1f}MW {r['tide_improvement']:>+6.1f}%")

if __name__ == '__main__':
    main()
