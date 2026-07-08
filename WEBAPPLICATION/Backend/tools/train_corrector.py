"""Train ARDRegression corrector on 2025 DLinear errors.

Produces: app/models/dlinear/intraday_corrector.pkl
"""
import json, os, sys, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.ml.dlinear_engine import DLinearEngine, FEATURE_COLS, _add_cyclical_features
from app.ml.intraday_corrector import IntradayCorrector, BATCH_FEATURES

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent.parent.parent  # LOADFORECASINGPROJECT
CSV_PATH = PROJECT / "data" / "ecg_actual_demand_clean_with_temp.csv"
MODEL_DIR = BASE.parent / "models" / "dlinear"
CKPT_PATH = MODEL_DIR / "intraday_corrector.pkl"

INPUT_WINDOW = 168
FORECAST_HORIZON = 24

# Train on 2025, test on 2026
TRAIN_YEARS = [2025]
device = "cpu"


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"] - 1, unit="h")
    df["hour_0_23"] = df["hour"] - 1
    df = df.sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["temperature_c"] = df["temp_c"].astype(float)
    return df


def generate_predictions(df, year, engine, label):
    """Generate DLinear raw predictions for all days in given year."""
    dates = sorted(df[df["year"] == year]["date"].unique())
    df_hist = _add_cyclical_features(df.copy())
    n = len(dates)
    t0 = time.time()

    for di, day_str in enumerate(dates):
        day_dt = pd.Timestamp(day_str)
        history_cutoff = day_dt - pd.Timedelta(hours=1)
        history = df_hist[df_hist["datetime"] <= history_cutoff].tail(INPUT_WINDOW + 24)
        if len(history) < INPUT_WINDOW:
            continue
        hist_df = pd.DataFrame({
            "date": pd.to_datetime(history["datetime"].values),
            "demand_mw": history["demand_mw"].values,
            "temperature_c": history["temperature_c"].values,
        })
        temps = df[(df["date"] == day_str) & (df["year"] == year)]["temperature_c"].tolist()
        if len(temps) < FORECAST_HORIZON:
            temps = temps + [28.0] * (FORECAST_HORIZON - len(temps))
        r = engine.predict(hist_df, horizon_hours=FORECAST_HORIZON,
                           future_temps_c=temps, use_tide=False)
        mask = (df["date"] == day_str) & (df["year"] == year)
        avail = min(mask.sum(), len(r["forecast_mw"]))
        if avail > 0:
            df.loc[mask, label] = r["forecast_mw"][:avail]

        engine.update(
            df.loc[mask, "demand_mw"].values[:avail],
            np.array(r["forecast_mw"][:avail]),
        )

        if (di + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{di+1}/{n}] {day_str} ({elapsed/(di+1):.1f}s/day)")

    elapsed = time.time() - t0
    print(f"  Done: {n} days in {elapsed:.0f}s ({elapsed/n:.1f}s/day avg)")
    return df


def build_features(df, year, fold_stats):
    """Build corrector features from DLinear errors.

    Only uses batch-available features (no lag features).
    """
    sub = df[df["year"] == year].copy()
    sub["hour_sin"] = np.sin(2 * np.pi * sub["hour_0_23"] / 24)
    sub["hour_cos"] = np.cos(2 * np.pi * sub["hour_0_23"] / 24)
    sub["dow_sin"] = np.sin(2 * np.pi * sub["dayofweek"] / 7)
    sub["dow_cos"] = np.cos(2 * np.pi * sub["dayofweek"] / 7)
    sub["month_sin"] = np.sin(2 * np.pi * sub["month"] / 12)
    sub["month_cos"] = np.cos(2 * np.pi * sub["month"] / 12)
    sub["weekend"] = sub["dayofweek"].isin([5, 6]).astype(int)

    sub["dlinear_error"] = sub["demand_mw"] - sub["dlinear_raw"]

    sub = sub.dropna().reset_index(drop=True)

    X = pd.DataFrame()
    for c in BATCH_FEATURES:
        X[c] = sub[c].values.astype(np.float32)

    y = sub["dlinear_error"].values.astype(np.float32)
    return X, y


def main():
    print("=" * 60)
    print("  Training ARDRegression Corrector")
    print("=" * 60)

    print("\nLoading data...")
    df = load_data()
    print(f"  {len(df)} rows, {df['date'].min()} to {df['date'].max()}")

    print("\nLoading engine...")
    engine = DLinearEngine(
        checkpoint_dir=str(MODEL_DIR),
        stats_path=str(MODEL_DIR / "normalization_stats.json"),
    )
    if not engine.is_fitted:
        print("ERROR: DLinear models not loaded!")
        sys.exit(1)

    # Get fold stats
    fold_key = list(engine.normalization_stats.keys())[-1]
    fold_stats = engine.normalization_stats[fold_key]
    print(f"  Using {fold_key} normalization: mean={fold_stats['means']['demand_mw']:.0f}, std={fold_stats['stds']['demand_mw']:.0f}")

    # Generate training predictions
    print(f"\nGenerating DLinear predictions for {TRAIN_YEARS}...")
    engine.reset_bias()
    df["dlinear_raw"] = np.nan
    df = generate_predictions(df, 2025, engine, "dlinear_raw")

    train_mask = df["year"].isin(TRAIN_YEARS) & df["dlinear_raw"].notna()
    print(f"\nTraining samples: {train_mask.sum():,} hours")

    print("\nBuilding features...")
    X, y = build_features(df, 2025, fold_stats)
    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  y stats: mean={float(np.mean(y)):.4f}, std={float(np.std(y)):.4f}")

    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    print("\nTraining ARDRegression...")
    corrector = IntradayCorrector()
    corrector.fit(X_scaled, y)
    print(f"  Residual std: {float(np.std(y - corrector._model.predict(X_scaled))):.4f} MW")

    # Feature weights
    coefs = corrector._model.coef_
    print("\nFeature weights:")
    for name, w in sorted(zip(BATCH_FEATURES, coefs), key=lambda x: -abs(x[1])):
        print(f"  {name:20s} {w:+.2f}")

    # Save
    print(f"\nSaving to {CKPT_PATH}...")
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CKPT_PATH, "wb") as f:
        pickle.dump({
            "model": corrector._model,
            "scaler": scaler,
            "feature_cols": BATCH_FEATURES,
            "residual_std": corrector._residual_std,
        }, f)
    print(f"  Saved ({os.path.getsize(CKPT_PATH) / 1024:.1f} KB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
