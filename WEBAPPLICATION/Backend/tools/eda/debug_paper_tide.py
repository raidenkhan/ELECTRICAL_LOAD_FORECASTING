"""
Verify: the paper's ACTUAL TIDE implementation (not our 24-hour-stride misinterpretation).
The paper's TIDE:
  - update(24h_pred, 24h_actual) each day
  - get_bias: mean error over last 48 hours (2 days), EMA-smoothed
  - SINGLE 24-element bias vector, NOT 24 independent hour biases
"""
import pandas as pd, numpy as np, sys, json
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
CSV_PATH = BASE / "tools" / "dl_forecast" / "ecg_demand_2018_2026.csv"
MODEL_DIR = BASE / "models" / "dlinear"
sys.path.insert(0, str(BASE))
from app.ml.dlinear_engine import DLinearEngine

# ── Paper's exact TIDE implementation ──
class PaperTide:
    """Matches the production _TideCorrector exactly."""
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
        errors = np.array([a - p for p, a in recent])
        bias = np.mean(errors, axis=0)
        if self._ema_bias is not None:
            self._ema_bias = self.alpha * bias + (1 - self.alpha) * self._ema_bias
        else:
            self._ema_bias = bias
        return self._ema_bias.copy()

    def apply(self, raw_pred):
        return raw_pred + self.get_bias()

    def update(self, prediction, actual):
        self.error_buffer.append((prediction.copy(), actual.copy()))

    def reset(self):
        self.error_buffer = []
        self._ema_bias = None

# ── Generate DLinear predictions ──
df = pd.read_csv(CSV_PATH)
df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"] - 1, unit="h")
df["hour_0_23"] = df["hour"] - 1
df = df.sort_values("datetime").reset_index(drop=True)

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
print(f"\nRaw DLinear MAE: {raw_mae:.1f} MW")
print(f"Raw DLinear bias: {test_clean['dlinear_error'].mean():.1f} MW")

# ── Run paper's TIDE (day-by-day, exactly as in production) ──
tide = PaperTide(alpha=0.3)
tide_errors = []
all_errors = test_clean["dlinear_error"].values
all_hours = test_clean["hour_0_23"].values
n_hours = len(all_errors)

# Simulate day-by-day: each day we get 24 predictions + 24 actuals
unique_dates = sorted(test_clean["date"].unique())
bias_over_time = []

for di, day in enumerate(unique_dates):
    mask = test_clean["date"] == day
    day_errors = all_errors[mask.values]
    day_preds = test_clean.loc[mask, "dlinear_raw"].values
    day_actuals = test_clean.loc[mask, "demand_mw"].values

    if len(day_errors) < 24:
        continue

    # Get correction BEFORE updating
    corr = tide.get_bias()  # 24-element bias vector (from previous days)
    corrected = day_preds + corr
    corr_errors = day_actuals - corrected
    tide_errors.extend(abs(corr_errors))

    # Update TIDE with today's data
    tide.update(day_preds, day_actuals)
    bias_over_time.append((day, corr.copy()))

tide_mae = np.mean(tide_errors)
print(f"Paper TIDE MAE: {tide_mae:.1f} MW")
print(f"Change: {(tide_mae - raw_mae) / raw_mae * 100:+.1f}%")

# ── Why? Check the bias vector over time ──
print(f"\nBias vector (last 3 days):")
for day, bias in bias_over_time[-3:]:
    print(f"  {day}: mean={np.mean(bias):+.1f} MW, std={np.std(bias):.1f} MW, range=[{np.min(bias):+.1f},{np.max(bias):+.1f}]")

# ── Compare: what if window was longer (30 days)? ──
print(f"\nTesting with longer window (30 days = 720 hours):")
tide30 = PaperTide(alpha=0.3, window_hours=720)
tide30_errors = []
for di, day in enumerate(unique_dates):
    mask = test_clean["date"] == day
    day_preds = test_clean.loc[mask, "dlinear_raw"].values
    day_actuals = test_clean.loc[mask, "demand_mw"].values
    if len(day_preds) < 24: continue
    corr = tide30.get_bias()
    corrected = day_preds + corr
    corr_errors = day_actuals - corrected
    tide30_errors.extend(abs(corr_errors))
    tide30.update(day_preds, day_actuals)

tide30_mae = np.mean(tide30_errors)
print(f"  Paper TIDE (30d window): {tide30_mae:.1f} MW ({((tide30_mae-raw_mae)/raw_mae*100):+.1f}%)")

# ── Compare: what if window was 90 days? ──
tide90 = PaperTide(alpha=0.3, window_hours=2160)
tide90_errors = []
for di, day in enumerate(unique_dates):
    mask = test_clean["date"] == day
    day_preds = test_clean.loc[mask, "dlinear_raw"].values
    day_actuals = test_clean.loc[mask, "demand_mw"].values
    if len(day_preds) < 24: continue
    corr = tide90.get_bias()
    corrected = day_preds + corr
    corr_errors = day_actuals - corrected
    tide90_errors.extend(abs(corr_errors))
    tide90.update(day_preds, day_actuals)

tide90_mae = np.mean(tide90_errors)
print(f"  Paper TIDE (90d window): {tide90_mae:.1f} MW ({((tide90_mae-raw_mae)/raw_mae*100):+.1f}%)")

# ── Compare with Lag-1 Direct ──
lag1_errors = []
prev_err = 0.0
for i in range(n_hours):
    err = all_errors[i]
    corr = prev_err if i > 0 else 0.0
    lag1_errors.append(abs(err - corr))
    prev_err = err
lag1_mae = np.mean(lag1_errors)
print(f"  Lag-1 Direct:         {lag1_mae:.1f} MW ({((lag1_mae-raw_mae)/raw_mae*100):+.1f}%)")

# ── Hour-by-hour bias of TIDE bias vector ──
print(f"\nHour-by-hour: TIDE bias vs actual error bias")
final_bias = tide._ema_bias
for h in range(24):
    actual_bias = np.mean(all_errors[all_hours == h])
    print(f"  Hour {h:2d}: TIDE bias={final_bias[h]:+.1f}  actual bias={actual_bias:+6.1f}  match={abs(final_bias[h]-actual_bias) < 10}")
