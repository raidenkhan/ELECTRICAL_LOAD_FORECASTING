"""
Ablation: Why do ARD/Bayesian beat KNN and Lag-1?
Checks: distribution shift, feature subsets, coefficient analysis, leakage.
"""
import pandas as pd
import numpy as np
import sys, json
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.metrics import mean_absolute_error

BASE = Path(__file__).parent.parent.parent
CSV_PATH = BASE / "tools" / "dl_forecast" / "ecg_demand_2018_2026.csv"
MODEL_DIR = BASE / "models" / "dlinear"
sys.path.insert(0, str(BASE))
from app.ml.dlinear_engine import DLinearEngine

df = pd.read_csv(CSV_PATH)
df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"] - 1, unit="h")
df["hour_0_23"] = df["hour"] - 1
df = df.sort_values("datetime").reset_index(drop=True)
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["dayofweek"] = df["datetime"].dt.dayofweek

# Generate DLinear preds
engine = DLinearEngine(checkpoint_dir=str(MODEL_DIR), stats_path=str(MODEL_DIR / "normalization_stats.json"),
                       db_path=str(BASE / "tools" / "eda" / "_ablation_tide.db"))

def generate_preds(target_df):
    dates = sorted(target_df["date"].unique())
    engine.reset_bias()
    for di, day_str in enumerate(dates):
        day_dt = pd.Timestamp(day_str)
        history = df[df["datetime"] <= day_dt - pd.Timedelta(hours=1)].tail(168 + 24)
        if len(history) < 168: continue
        hist_df = pd.DataFrame({
            "date": pd.to_datetime(history["datetime"].values),
            "demand_mw": history["demand_mw"].values,
            "temperature_c": history["temperature_c"].values,
        })
        day_temps = target_df[target_df["date"] == day_str]["temperature_c"].tolist()
        if len(day_temps) < 24:
            day_temps = day_temps + [28.0] * (24 - len(day_temps))
        r_raw = engine.predict(hist_df, horizon_hours=24, future_temps_c=day_temps, use_tide=False)
        day_actuals = target_df[target_df["date"] == day_str]["demand_mw"].values
        if len(day_actuals) < 24: continue
        mask = target_df["date"] == day_str
        target_df.loc[mask, "dlinear_raw"] = r_raw["forecast_mw"]
    return target_df

df_train = df[df["datetime"].dt.year == 2025].copy()
df_train["dlinear_raw"] = np.nan
df_train = generate_preds(df_train)

df_test = df[df["datetime"].dt.year == 2026].copy()
df_test["dlinear_raw"] = np.nan
df_test = generate_preds(df_test)

# ── Features (temporally clean, same as main ablation) ──
def add_features(d):
    d = d.copy()
    d["hour_sin"] = np.sin(2 * np.pi * d["hour_0_23"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour_0_23"] / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["dayofweek"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["dayofweek"] / 7)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["weekend"] = d["dayofweek"].isin([5, 6]).astype(int)
    d["days_from_start"] = (d["datetime"] - df["datetime"].min()).dt.days
    d["dlinear_error"] = d["demand_mw"] - d["dlinear_raw"]
    d["ramp_1h"] = d["demand_mw"].diff(1).shift(1)
    d["ramp_3h"] = d["demand_mw"].diff(3).shift(1) / 3
    d["rolling_err_6h"] = d["dlinear_error"].shift(1).rolling(6, min_periods=2).mean()
    d["rolling_abs_err_6h"] = d["dlinear_error"].abs().shift(1).rolling(6, min_periods=2).mean()
    for lag in [1, 2, 3, 24]:
        d[f"err_lag_{lag}"] = d["dlinear_error"].shift(lag)
    d["temperature_c"] = d["temperature_c"]
    return d

FEATURE_SETS = {
    "err_lag_1_only": ["err_lag_1"],
    "cyclical_only": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "weekend"],
    "no_err_lags": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
                    "weekend", "ramp_1h", "ramp_3h", "temperature_c", "days_from_start"],
    "err_lag1_temp": ["err_lag_1", "temperature_c"],
    "err_lag1_temp_ramp": ["err_lag_1", "temperature_c", "ramp_1h", "ramp_3h"],
    "full": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
             "weekend", "ramp_1h", "ramp_3h", "rolling_err_6h", "rolling_abs_err_6h",
             "err_lag_1", "err_lag_2", "err_lag_3", "err_lag_24", "temperature_c", "days_from_start"],
}

# ── 1. Check distribution shift: 2025 vs 2026 features ──
print("=" * 70)
print("PART 1: DISTRIBUTION SHIFT CHECK")
print("=" * 70)

train = add_features(df_train.dropna(subset=["dlinear_raw"]))
test = add_features(df_test.dropna(subset=["dlinear_raw"]))
for col in ["err_lag_1", "err_lag_24", "rolling_err_6h", "dlinear_error"]:
    t_mean = train[col].mean()
    t_std = train[col].std()
    e_mean = test[col].mean()
    e_std = test[col].std()
    shift_std = abs(t_mean - e_mean) / ((t_std + e_std) / 2)
    print(f"  {col:20s}: train mean={t_mean:+7.2f} std={t_std:6.1f} | test mean={e_mean:+7.2f} std={e_std:6.1f} | shift={shift_std:.2f}s")

# ── 2. Distribution of error at each hour ──
print(f"\n--- Hourly error bias ratio (bias/MAE) ---")
for h in range(24):
    for label, d in [("2025", train), ("2026", test)]:
        m = d["hour_0_23"] == h
        eh = d.loc[m, "dlinear_error"]
        if len(eh) > 0 and label == "2025":
            r = abs(eh.mean()) / eh.abs().mean()
            print(f"  Hour {h:2d}: 2025 bias/MAE={r:.2f}", end=" | ")
        elif len(eh) > 0 and label == "2026":
            r = abs(eh.mean()) / eh.abs().mean()
            print(f"2026 bias/MAE={r:.2f}")

# ── 3. Feature ablation: ARD, BayesianRidge, KNN ──
print("\n" + "=" * 70)
print("PART 2: MODEL x FEATURE-SET ABLATION")
print("=" * 70)

train_clean = train.dropna()
test_clean = test.dropna().reset_index(drop=True)

models = [
    ("KNN (k=10)", lambda: KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)),
    ("BayesianRidge", lambda: BayesianRidge(compute_score=True, tol=1e-4)),
    ("ARDRegression",  lambda: ARDRegression(compute_score=True, tol=1e-4)),
]

for fset_name, fset_cols in FEATURE_SETS.items():
    print(f"\n  Features: {fset_name}")
    print(f"  {'Model':20s} | {'Train MAE':>9s} | {'Test MAE':>8s} | {'chg%':>6s}")
    print(f"  {'-'*20}-+-{'-'*9}-+-{'-'*8}-+-{'-'*6}")
    for mname, mfactory in models:
        X_tr = train_clean[fset_cols].values
        y_tr = train_clean["dlinear_error"].values
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        model = mfactory()
        model.fit(X_tr_s, y_tr)
        train_pred = model.predict(X_tr_s)
        train_mae = mean_absolute_error(y_tr, train_pred)

        X_te = test_clean[fset_cols].values
        y_te = test_clean["dlinear_error"].values
        X_te_s = scaler.transform(X_te)
        test_pred = model.predict(X_te_s)
        test_mae = mean_absolute_error(y_te, test_pred)

        raw_mae = mean_absolute_error(y_te, np.zeros_like(y_te))
        imp = (raw_mae - test_mae) / raw_mae * 100
        print(f"  {mname:20s} | {train_mae:>8.1f} | {test_mae:>7.1f} | {imp:>+5.1f}%")

# ── 4. ARD coefficient analysis ──
print("\n" + "=" * 70)
print("PART 3: ARD FEATURE WEIGHTS")
print("=" * 70)

X_tr = train_clean[FEATURE_SETS["full"]].values
y_tr = train_clean["dlinear_error"].values
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
ard = ARDRegression(compute_score=True, tol=1e-4)
ard.fit(X_tr_s, y_tr)
coefs = list(zip(FEATURE_SETS["full"], ard.coef_))
coefs.sort(key=lambda x: abs(x[1]), reverse=True)
for feat, coef in coefs:
    print(f"  {feat:20s}: {coef:+8.4f}")

# ── 5. Leakage check: what if we use the CURRENT row's err_lag_1? ──
# If there was leakage, err_lag_1 would correlate with the error because
# err_lag_1 uses the CURRENT error! But we use shift(1), so it's the PAST error.
# Quick sanity: simulate Lag-1 Direct on test, but use err_lag_1 from features
print("\n" + "=" * 70)
print("PART 4: LEAKAGE VERIFICATION")
print("=" * 70)
sim_raw = []
sim_lag1 = []
for idx, row in test_clean.iterrows():
    err = row["dlinear_error"]
    sim_raw.append(abs(err))
    sim_lag1.append(abs(err - row["err_lag_1"]))

print(f"  Raw MAE:           {np.mean(sim_raw):.1f}")
print(f"  Lag-1 Direct:      {np.mean(sim_lag1):.1f}")
print(f"  (should match Lag-1 Direct in main ablation = ~72.5)")

# Check: err_lag_1 on test is ACTUALLY the previous hour's error?
print(f"\n  err_lag_1 == shift(1) of dlinear_error?")
print(f"  Test: row[1]['err_lag_1'] = {test_clean.iloc[1]['err_lag_1']:.1f}")
print(f"  vs row[0]['dlinear_error'] = {test_clean.iloc[0]['dlinear_error']:.1f}")
print(f"  Match: {'YES - No leakage' if abs(test_clean.iloc[1]['err_lag_1'] - test_clean.iloc[0]['dlinear_error']) < 0.01 else 'SUSPICIOUS'}")

# ── 6. What makes ARD beat Lag-1? — predict error on subset of hours ──
print("\n" + "=" * 70)
print("PART 5: WHERE DOES ARD HELP OVER LAG-1?")
print("=" * 70)
err_lag1_pred = test_clean["err_lag_1"].values
ard_pred = model.predict(X_te_s)
for h in [0, 6, 12, 18, 23]:
    mask = test_clean["hour_0_23"] == h
    raw = mean_absolute_error(test_clean.loc[mask, "dlinear_error"], np.zeros(mask.sum()))
    lag1 = mean_absolute_error(test_clean.loc[mask, "dlinear_error"], err_lag1_pred[mask])
    ard_mae = mean_absolute_error(test_clean.loc[mask, "dlinear_error"], ard_pred[mask])
    print(f"  Hour {h:2d}: raw={raw:6.1f}  lag1={lag1:6.1f}  ard={ard_mae:6.1f}  (ard chg={((ard_mae-lag1)/lag1*100):+.1f}%)")
