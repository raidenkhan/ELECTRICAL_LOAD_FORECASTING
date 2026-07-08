"""
Test: What if TIDE tracked the previous hour's error instead of hour-specific bias?
Compare:
  1. Original TIDE: 24 separate hour-biases
  2. Lag-TIDE: Single EMA of most recent error (no hour-state)
  3. Lag-1 direct: correction = error[t-1]
  4. Lag-1 dampened: correction = alpha * error[t-1]
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
CSV_PATH = BASE / "tools" / "dl_forecast" / "ecg_demand_2018_2026.csv"
MODEL_DIR = BASE / "models" / "dlinear"
sys.path.insert(0, str(BASE))
from app.ml.dlinear_engine import DLinearEngine

df = pd.read_csv(CSV_PATH)
df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"] - 1, unit="h")
df = df.sort_values("datetime").reset_index(drop=True)

df_test = df[df["datetime"].dt.year == 2026].copy()
df_test["dlinear_raw"] = np.nan

engine = DLinearEngine(
    checkpoint_dir=str(MODEL_DIR),
    stats_path=str(MODEL_DIR / "normalization_stats.json"),
    db_path=str(BASE / "tools" / "eda" / "_ablation_tide.db"),
)

dates = sorted(df_test["date"].unique())
print(f"Generating predictions for {len(dates)} days...")
for di, day_str in enumerate(dates):
    day_dt = pd.Timestamp(day_str)
    history_cutoff = day_dt - pd.Timedelta(hours=1)
    history = df[df["datetime"] <= history_cutoff].tail(168 + 24)
    if len(history) < 168:
        continue
    hist_df = pd.DataFrame({
        "date": pd.to_datetime(history["datetime"].values),
        "demand_mw": history["demand_mw"].values,
        "temperature_c": history["temperature_c"].values,
    })
    day_temps = df_test[df_test["date"] == day_str]["temperature_c"].tolist()
    if len(day_temps) < 24:
        day_temps = day_temps + [28.0] * (24 - len(day_temps))
    r_raw = engine.predict(hist_df, horizon_hours=24, future_temps_c=day_temps, use_tide=False)
    day_actuals = df_test[df_test["date"] == day_str]["demand_mw"].values
    if len(day_actuals) < 24:
        continue
    mask = df_test["date"] == day_str
    df_test.loc[mask, "dlinear_raw"] = r_raw["forecast_mw"]

df_test["dlinear_error"] = df_test["demand_mw"] - df_test["dlinear_raw"]

# ── Simulation ──
results = df_test.dropna(subset=["dlinear_error"]).reset_index(drop=True)

def simulate_tide(tide_type, alpha=0.3):
    """Simulate different TIDE variants."""
    errors = []
    if tide_type == "hour_bias":
        # Original: 24 hour-specific biases, updated once per day per hour
        bias = np.zeros(24)
        counts = np.zeros(24)
    elif tide_type == "lag_ema":
        # Single EMA tracking the most recent error
        bias = 0.0
        count = 0
    elif tide_type == "lag1_direct":
        # Just use previous error directly
        prev_err = 0.0
        has_prev = False
    elif tide_type == "lag1_dampened":
        prev_err = 0.0
        has_prev = False

    for idx, row in results.iterrows():
        a = row["demand_mw"]
        p = row["dlinear_raw"]
        err = a - p

        if tide_type == "hour_bias":
            h = int(row["hour"]) - 1
            corr = bias[h] if counts[h] > 0 else 0.0
            corrected_err = err - corr
            if counts[h] == 0:
                bias[h] = err
            else:
                bias[h] = alpha * err + (1 - alpha) * bias[h]
            counts[h] += 1

        elif tide_type == "lag_ema":
            corr = bias if count > 0 else 0.0
            corrected_err = err - corr
            if count == 0:
                bias = err
            else:
                bias = alpha * err + (1 - alpha) * bias
            count += 1

        elif tide_type == "lag1_direct":
            corr = prev_err if has_prev else 0.0
            corrected_err = err - corr
            prev_err = err
            has_prev = True

        elif tide_type == "lag1_dampened":
            corr = alpha * prev_err if has_prev else 0.0
            corrected_err = err - corr
            prev_err = err
            has_prev = True

        errors.append(abs(corrected_err))

    return float(np.mean(errors))

raw_mae = float(np.mean(np.abs(results["dlinear_error"].values)))

print(f"\nRaw DLinear MAE: {raw_mae:.1f} MW")
print(f"\n{'Variant':25s} | MAE    | chg%")
print("-" * 45)
for name, fn in [("Hour-Bias TIDE (original)", "hour_bias"),
                  ("Lag-EMA TIDE (single)", "lag_ema"),
                  ("Lag-1 Direct (err[t-1])", "lag1_direct"),
                  ("Lag-1 Dampened (α×err[t-1])", "lag1_dampened")]:
    mae = simulate_tide(fn)
    imp = (mae - raw_mae) / raw_mae * 100
    print(f"{name:25s} | {mae:5.1f} | {imp:+5.1f}%")

# Also test: what if we use err_lag_1 with a learned coefficient?
print(f"\nLag-1 correlation (r): {np.corrcoef(results['dlinear_error'].values[:-1], results['dlinear_error'].values[1:])[0,1]:+.4f}")
