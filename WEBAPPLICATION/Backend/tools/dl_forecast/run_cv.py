"""Run 6-fold cross-validation for all deep learning models.

Usage:
    python run_cv.py [--models lstm gru transformer tcn dlinear] [--batch_size 4096]

Output:
    Creates results/ directory with:
    - results.csv: per-fold, per-model MAE
    - results_summary.csv: mean +/- std across folds
    - {model}_fold_{n}_best.pt: best model checkpoints
    - per_hour_mae_{model}.csv: per-hour MAE averaged across folds
"""
import argparse, os, sys, time, gc, json
import pandas as pd
import numpy as np
import torch

from config import Config
from data import load_and_prepare_data, normalize_features, make_dataloader, FEATURE_COLS
from models import create_model, count_parameters, MODEL_REGISTRY
from train import train_model, validate


def run_fold(model_name, cfg, df_full, train_dates, test_dates, device, results_dir):
    fold_name = f"Fold_{str(train_dates[1])[:4]}_to_{str(test_dates[1])[:4]}"
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} | {fold_name}")
    print(f"{'='*60}", flush=True)

    t_start = time.time()

    # Split
    tr_mask = (df_full['date'] >= train_dates[0]) & (df_full['date'] <= train_dates[1])
    te_mask = (df_full['date'] >= test_dates[0]) & (df_full['date'] <= test_dates[1])
    df_tr = df_full[tr_mask].copy()
    df_te = df_full[te_mask].copy()

    # Normalize per-fold using ONLY training data statistics (no leakage)
    df_tr, fold_means, fold_stds = normalize_features(df_tr, FEATURE_COLS)
    df_te, _, _ = normalize_features(df_te, FEATURE_COLS, fold_means, fold_stds)

    n_features = len(FEATURE_COLS)

    train_loader = make_dataloader(
        df_tr, cfg.input_window, cfg.forecast_horizon, cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
    )
    val_loader = make_dataloader(
        df_te, cfg.input_window, cfg.forecast_horizon, cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
    )

    # Print dataset sizes for debugging
    print(f"  Train samples: {len(train_loader.dataset):,} | Test samples: {len(val_loader.dataset):,}", flush=True)

    # Build model
    model = create_model(model_name, n_features, cfg.forecast_horizon, cfg.input_window, cfg)
    model = model.to(device)
    param_count = count_parameters(model)
    print(f"  Parameters: {param_count:,}", flush=True)

    # Count batches for gradient accumulation estimation
    batches_per_epoch = len(train_loader)
    print(f"  Batches/epoch: {batches_per_epoch:,} | Batch size: {cfg.batch_size}", flush=True)

    # Train
    model, best_val_mae, per_hour_mae = train_model(model, train_loader, val_loader, cfg, device)

    elapsed = time.time() - t_start

    # Save checkpoint
    ckpt_path = os.path.join(results_dir, f"{model_name}_fold_{fold_name}_best.pt")
    torch.save({
        'model_state': model.state_dict(),
        'val_mae': best_val_mae,
        'per_hour_mae': per_hour_mae,
    }, ckpt_path)
    print(f"  Saved: {ckpt_path}", flush=True)

    # Per-hour breakdown
    early = np.mean(per_hour_mae[:8])
    mid = np.mean(per_hour_mae[8:16])
    late = np.mean(per_hour_mae[16:])

    result = {
        'model': model_name,
        'fold': fold_name,
        'mae_24h': float(best_val_mae),
        'mae_early': float(early),
        'mae_mid': float(mid),
        'mae_late': float(late),
        'params': param_count,
        'train_time_s': elapsed,
        'train_samples': len(train_loader.dataset),
        'test_samples': len(val_loader.dataset),
    }
    print(f"  Result: {best_val_mae:.0f} MW ({elapsed:.0f}s)", flush=True)
    return result, per_hour_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Models to train')
    parser.add_argument('--data_path', type=str, default='ecg_demand_2018_2026.csv')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--ckpt', action='store_true', help='Save full model checkpoints (large files)')
    args = parser.parse_args()

    cfg = Config()
    if args.batch_size:
        cfg.batch_size = args.batch_size

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, args.data_path)
    if not os.path.exists(data_path):
        data_path = args.data_path
    if not os.path.exists(data_path):
        print(f"ERROR: data file not found at {data_path}")
        sys.exit(1)

    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        print(f"CUDA version: {torch.version.cuda}")

    # Load data once
    print("\nLoading data...", flush=True)
    df = load_and_prepare_data(data_path)
    print(f"  {len(df):,} rows | Date range: {df['date'].min()} to {df['date'].max()}", flush=True)

    # All models use the same base df; normalization is done per-fold (no leakage)
    print(f"  Features: {FEATURE_COLS}", flush=True)

    all_results = []
    per_hour_results = {}  # model_name -> list of per_hour_mae arrays

    for model_name in args.models:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"\n{'#'*60}")
        print(f"#  MODEL: {model_name.upper()}")
        print(f"{'#'*60}", flush=True)
        t_global = time.time()

        model_results = []
        model_per_hour = []

        for fold_name, tr_s, tr_e, te_s, te_e in cfg.folds:
            train_dates = (pd.to_datetime(tr_s), pd.to_datetime(tr_e))
            test_dates = (pd.to_datetime(te_s), pd.to_datetime(te_e))

            result, per_hour = run_fold(
                model_name, cfg, df, train_dates, test_dates,
                device, results_dir,
            )
            model_results.append(result)
            model_per_hour.append(per_hour)

        total_time = time.time() - t_global

        # Model summary
        maes = [r['mae_24h'] for r in model_results]
        early_maes = [r['mae_early'] for r in model_results]
        mid_maes = [r['mae_mid'] for r in model_results]
        late_maes = [r['mae_late'] for r in model_results]

        print(f"\n  {model_name.upper()} 6-FOLD CV SUMMARY:")
        print(f"  {'Fold':<12} {'MAE':>8} {'Early':>8} {'Mid':>8} {'Late':>8}")
        for r in model_results:
            print(f"  {r['fold']:<12} {r['mae_24h']:>6.0f}MW {r['mae_early']:>6.0f}MW {r['mae_mid']:>6.0f}MW {r['mae_late']:>6.0f}MW")
        print(f"  {'Mean':<12} {np.mean(maes):>6.0f}MW {np.mean(early_maes):>6.0f}MW {np.mean(mid_maes):>6.0f}MW {np.mean(late_maes):>6.0f}MW")
        print(f"  {'Std':<12} {np.std(maes):>6.0f}MW {np.std(early_maes):>6.0f}MW {np.std(mid_maes):>6.0f}MW {np.std(late_maes):>6.0f}MW")
        print(f"  Best: {min(maes):.0f} MW | Worst: {max(maes):.0f} MW | Time: {total_time:.0f}s")
        print(f"  Wins vs adaptive (119 MW): {sum(1 for m in maes if m < 119)}/6", flush=True)

        all_results.extend(model_results)
        per_hour_results[model_name] = np.array(model_per_hour)

    # ── Global Summary ──
    print(f"\n{'='*70}")
    print(f"  GLOBAL CROSS-MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<15} {'Mean MAE':>10} {'Std':>7} {'Best':>7} {'Worst':>8} {'Params':>10} {'Wins/6'}")
    print(f"  {'-'*65}")
    
    for model_name in args.models:
        m_results = [r for r in all_results if r['model'] == model_name]
        maes = [r['mae_24h'] for r in m_results]
        params = m_results[0]['params'] if m_results else 0
        wins = sum(1 for m in maes if m < 119)
        print(f"  {model_name:<15} {np.mean(maes):>6.0f}MW {np.std(maes):>5.0f}MW {min(maes):>5.0f}MW {max(maes):>6.0f}MW {params:>9,} {wins:>3}/6")

    print(f"  {'Adaptive':<15} {'119':>6}MW {'29':>5}MW {'83':>5}MW {'148':>6}MW", flush=True)

    # Save global results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(results_dir, 'results.csv'), index=False)
    
    # Per-hour MAE summary (average across folds)
    for model_name, per_hour_arr in per_hour_results.items():
        mean_per_hour = np.mean(per_hour_arr, axis=0)
        np.savetxt(os.path.join(results_dir, f'per_hour_mae_{model_name}.csv'),
                   mean_per_hour, delimiter=',',
                   header=','.join(str(h+1) for h in range(24)),
                   comments='')

    print(f"\n  Results saved to {results_dir}/")
    print(f"  DONE", flush=True)


if __name__ == '__main__':
    main()
