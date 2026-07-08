"""
Debug: Why does TIDE hurt DLinear? Check error autocorrelation and hour-by-hour bias.
"""
import pandas as pd
import numpy as np
import sys, json
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
CSV_PATH = BASE / "tools" / "dl_forecast" / "ecg_demand_2018_2026.csv"
MODEL_DIR = BASE / "models" / "dlinear"
sys.path.insert(0, str(BASE))
from app.ml.dlinear_engine import DLinearEngine

df = pd.read_csv(CSV_PATH)
df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"] - 1, unit="h")
df = df.sort_values("datetime").reset_index(drop=True)

# Generate DLinear predictions for 2026
df_test = df[df["datetime"].dt.year == 2026].copy()
df_test["dlinear_raw"] = np.nan

engine = DLinearEngine(
    checkpoint_dir=str(MODEL_DIR),
    stats_path=str(MODEL_DIR / "normalization_stats.json"),
    db_path=str(BASE / "tools" / "eda" / "_ablation_tide.db"),
)

INPUT_WINDOW = 168
FORECAST_HORIZON = 24

dates = sorted(df_test["date"].unique())
print(f"Generating predictions for {len(dates)} days...")
for di, day_str in enumerate(dates):
    day_dt = pd.Timestamp(day_str)
    history_cutoff = day_dt - pd.Timedelta(hours=1)
    history = df[df["datetime"] <= history_cutoff].tail(INPUT_WINDOW + 24)
    if len(history) < INPUT_WINDOW:
        continue
    hist_df = pd.DataFrame({
        "date": pd.to_datetime(history["datetime"].values),
        "demand_mw": history["demand_mw"].values,
        "temperature_c": history["temperature_c"].values,
    })
    day_temps = df_test[df_test["date"] == day_str]["temperature_c"].tolist()
    if len(day_temps) < FORECAST_HORIZON:
        day_temps = day_temps + [28.0] * (FORECAST_HORIZON - len(day_temps))
    r_raw = engine.predict(hist_df, horizon_hours=FORECAST_HORIZON, future_temps_c=day_temps, use_tide=False)
    day_actuals = df_test[df_test["date"] == day_str]["demand_mw"].values
    if len(day_actuals) < FORECAST_HORIZON:
        continue
    mask = df_test["date"] == day_str
    df_test.loc[mask, "dlinear_raw"] = r_raw["forecast_mw"]

df_test["dlinear_error"] = df_test["demand_mw"] - df_test["dlinear_raw"]
df_test["hour_0_23"] = df_test["hour"] - 1
errors = df_test["dlinear_error"].dropna().values

print(f"\n=== DLinear Error Statistics (2026, {len(errors)} hours) ===")
print(f"Mean bias: {np.mean(errors):+.2f} MW")
print(f"MAE: {np.mean(np.abs(errors)):.2f} MW")
print(f"Std: {np.std(errors):.2f} MW")

print(f"\n=== Hour-by-hour bias ===")
for h in range(24):
    mask = df_test["hour_0_23"] == h
    eh = df_test.loc[mask, "dlinear_error"].dropna()
    if len(eh) == 0:
        continue
    print(f"  Hour {h:2d}: bias={np.mean(eh):+6.1f} MW, MAE={np.mean(np.abs(eh)):.1f} MW, count={len(eh)}")

# Autocorrelation
from scipy import stats
print(f"\n=== Error Autocorrelation ===")
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    if len(errors) <= lag:
        continue
    ac = np.corrcoef(errors[:-lag], errors[lag:])[0, 1]
    print(f"  Lag-{lag:3d}: r = {ac:+.4f}")

# Simulate simple TIDE
print(f"\n=== TIDE Simulation ===")
bias = np.zeros(24)
counts = np.zeros(24)
tide_errors = []
raw_errors = []
for idx, row in df_test.dropna(subset=["dlinear_error"]).iterrows():
    h = int(row["hour_0_23"])
    err = row["dlinear_error"]
    corr = bias[h] if counts[h] > 0 else 0.0
    corrected_err = err - corr
    tide_errors.append(abs(corrected_err))
    raw_errors.append(abs(err))
    if counts[h] == 0:
        bias[h] = err
    else:
        bias[h] = 0.3 * err + 0.7 * bias[h]
    counts[h] += 1

print(f"  Raw MAE:  {np.mean(raw_errors):.1f} MW")
print(f"  TIDE MAE: {np.mean(tide_errors):.1f} MW")
print(f"  Change:   {(np.mean(tide_errors)-np.mean(raw_errors))/np.mean(raw_errors)*100:+.1f}%")
