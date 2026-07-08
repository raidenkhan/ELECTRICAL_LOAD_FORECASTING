"""
Test the paper's ACTUAL TIDE correctly: in normalized z-score space.
"""
import pandas as pd, numpy as np, sys, json
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
CSV_PATH = BASE / "tools" / "dl_forecast" / "ecg_demand_2018_2026.csv"
MODEL_DIR = BASE / "models" / "dlinear"
sys.path.insert(0, str(BASE))
from app.ml.dlinear_engine import DLinearEngine

# ── Paper's TIDE (exact production implementation) ──
class PaperTide:
    def __init__(self, alpha=0.3, window_hours=48):
        self.alpha = alpha
        self.window_hours = window_hours
        self.error_buffer = []
        self._ema_bias = None

    def get_bias(self):
        if len(self.error_buffer) < 1:
            return np.zeros(24)
        recent = self.error_buffer[-self.window_hours // 24:]
        if len(recent) == 0:
            return np.zeros(24)
        errors = np.array([a - p for p, a in recent])  # (N, 24) in normalized space
        bias = np.mean(errors, axis=0)  # (24,) — mean per hour over N days
        if self._ema_bias is not None:
            self._ema_bias = self.alpha * bias + (1 - self.alpha) * self._ema_bias
        else:
            self._ema_bias = bias
        return self._ema_bias.copy()

    def apply(self, raw_pred, demand_std):
        """Apply correction to raw MW prediction using normalized bias."""
        bias_norm = self.get_bias()
        return raw_pred + bias_norm * demand_std

    def update(self, norm_pred, norm_actual):
        """Update with normalized (z-score) values."""
        self.error_buffer.append((norm_pred.copy(), norm_actual.copy()))

    def reset(self):
        self.error_buffer = []
        self._ema_bias = None

# ── Generate DLinear predictions ──
df = pd.read_csv(CSV_PATH)
df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"] - 1, unit="h")
df["hour_0_23"] = df["hour"] - 1
df = df.sort_values("datetime").reset_index(drop=True)

# Load normalization stats (Fold_6 mean/std)
with open(MODEL_DIR / "normalization_stats.json") as f:
    stats = json.load(f)
# Get last fold's stats (Fold_6)
last_fold = sorted(stats.keys())[-1] if isinstance(stats, dict) else "Fold_6"
fold_stats = stats[last_fold] if isinstance(stats, dict) else stats

demand_mean = np.float64(fold_stats["means"]["demand_mw"])
demand_std = np.float64(fold_stats["stds"]["demand_mw"])
print(f"Normalization stats: mean={demand_mean:.0f}, std={demand_std:.0f}")

df_test = df[df["datetime"].dt.year == 2026].copy()
df_test["dlinear_raw"] = np.nan

engine = DLinearEngine(checkpoint_dir=str(MODEL_DIR),
    stats_path=str(MODEL_DIR / "normalization_stats.json"),
    db_path=str(BASE / "tools" / "eda" / "_ablation_tide.db"))

dates = sorted(df_test["date"].unique())
print(f"Generating DLinear predictions for {len(dates)} days...")
for di, day_str in enumerate(dates):
    day_dt = pd.Timestamp(day_str)
    history = df[df["datetime"] <= day_dt - pd.Timedelta(hours=1)].tail(192)
    if len(history) < 168: continue
    hist_df = pd.DataFrame({
        "date": pd.to_datetime(history["datetime"].values),
        "demand_mw": history["demand_mw"].values,
        "temperature_c": history["temperature_c"].values,
    })
    day_temps = df_test[df_test["date"] == day_str]["temperature_c"].tolist()
    if len(day_temps) < 24: day_temps += [28.0] * (24 - len(day_temps))
    r = engine.predict(hist_df, horizon_hours=24, future_temps_c=day_temps, use_tide=False)
    actuals_d = df_test[df_test["date"] == day_str]["demand_mw"].values
    if len(actuals_d) < 24: continue
    df_test.loc[df_test["date"] == day_str, "dlinear_raw"] = r["forecast_mw"]

test_clean = df_test.dropna(subset=["dlinear_raw"]).reset_index(drop=True)
test_clean["dlinear_error"] = test_clean["demand_mw"] - test_clean["dlinear_raw"]

raw_mae = np.mean(np.abs(test_clean["dlinear_error"].values))
raw_bias = test_clean["dlinear_error"].mean()
print(f"\nRaw DLinear MAE: {raw_mae:.1f} MW, Bias: {raw_bias:+.1f} MW")

# ── Run paper's TIDE correctly (in normalized space) ──
tide = PaperTide(alpha=0.3)
tide_errors = []
unique_dates = sorted(test_clean["date"].unique())

for di, day in enumerate(unique_dates):
    mask = test_clean["date"] == day
    day_preds = test_clean.loc[mask, "dlinear_raw"].values
    day_actuals = test_clean.loc[mask, "demand_mw"].values
    if len(day_preds) < 24: continue

    # Apply correction using normalized bias
    corr = tide.get_bias()  # in normalized space
    correction_mw = corr * demand_std
    corrected = day_preds + correction_mw
    corr_errors = day_actuals - corrected
    tide_errors.extend(abs(corr_errors))

    # Update TIDE with NORMALIZED values
    norm_pred = (day_preds - demand_mean) / demand_std
    norm_actual = (day_actuals - demand_mean) / demand_std
    tide.update(norm_pred, norm_actual)

tide_mae = np.mean(tide_errors)
print(f"Paper TIDE (normalized): {tide_mae:.1f} MW ({((tide_mae-raw_mae)/raw_mae*100):+.1f}%)")

# ── Also test what the paper calls "live TIDE" via engine.predict(use_tide=True) ──
print(f"\nCompare with engine's live TIDE:")
# Re-run with use_tide=True
engine.reset_bias()
tide_live_errors = []
tide_live_raw = []
for di, day in enumerate(unique_dates):
    day_dt = pd.Timestamp(day)
    history = df[df["datetime"] <= day_dt - pd.Timedelta(hours=1)].tail(192)
    if len(history) < 168: continue
    hist_df = pd.DataFrame({
        "date": pd.to_datetime(history["datetime"].values),
        "demand_mw": history["demand_mw"].values,
        "temperature_c": history["temperature_c"].values,
    })
    day_temps = test_clean[test_clean["date"] == day]["temperature_c"].tolist()
    if len(day_temps) < 24: day_temps += [28.0] * (24 - len(day_temps))
    r = engine.predict(hist_df, horizon_hours=24, future_temps_c=day_temps, use_tide=True)
    actuals_d = test_clean[test_clean["date"] == day]["demand_mw"].values
    if len(actuals_d) < 24: continue
    tide_live_errors.extend(abs(actuals_d - r["forecast_mw"]))
    engine.update(actuals_d, np.array(r["forecast_mw"]))

live_mae = np.mean(tide_live_errors)
print(f"  Engine live TIDE: {live_mae:.1f} MW ({((live_mae-raw_mae)/raw_mae*100):+.1f}%)")

# ── What if we try a longer window for stabler bias estimate? ──
print(f"\nTIDE with longer windows (normalized space):")
for window_days in [2, 7, 14, 30, 60]:
    tide_w = PaperTide(alpha=0.3, window_hours=window_days * 24)
    tide_w_errors = []
    for di, day in enumerate(unique_dates):
        mask = test_clean["date"] == day
        day_preds = test_clean.loc[mask, "dlinear_raw"].values
        day_actuals = test_clean.loc[mask, "demand_mw"].values
        if len(day_preds) < 24: continue
        corr = tide_w.get_bias()
        corrected = day_preds + corr * demand_std
        tide_w_errors.extend(abs(day_actuals - corrected))
        norm_pred = (day_preds - demand_mean) / demand_std
        norm_actual = (day_actuals - demand_mean) / demand_std
        tide_w.update(norm_pred, norm_actual)
    w_mae = np.mean(tide_w_errors)
    print(f"  {window_days:2d}-day window: {w_mae:.1f} MW ({((w_mae-raw_mae)/raw_mae*100):+.1f}%)")

# ── What if we use the bias stats from the paper (simulate paper's conditions)? ──
print(f"\n=== Simulating paper's conditions ===")
print(f"Paper: bias=-18.4 MW, MAE=100.8, our model: bias=+3.9, MAE=115.6")
bias_diff = 18.4  # paper's bias magnitude
artificial_std_ratio = demand_std / np.std(test_clean["dlinear_error"].values)
print(f"Error std (normalized): {np.std(test_clean['dlinear_error'].values)/demand_std:.3f}")

# ── Hour-by-hour: final TIDE bias vs actual ──
print(f"\nHour-by-hour bias check (final TIDE bias vector):")
all_errors = test_clean["dlinear_error"].values
all_hours = test_clean["hour_0_23"].values
final_bias = tide._ema_bias
matches = 0
for h in range(24):
    actual_bias = np.mean(all_errors[all_hours == h]) / demand_std
    bias_norm = final_bias[h]
    diff = abs(bias_norm - actual_bias)
    match = diff < 0.02  # within 0.02 z-score
    if match: matches += 1
    print(f"  Hour {h:2d}: TIDE bias={bias_norm:+.4f} actual={actual_bias:+7.4f}  {'✓' if match else '✗'}")
print(f"\n  TIDE bias matches actual in {matches}/24 hours")
