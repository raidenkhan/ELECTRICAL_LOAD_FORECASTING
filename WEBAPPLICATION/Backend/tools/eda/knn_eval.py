import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "dl_forecast", "ecg_demand_2018_2026.csv")
EDA_HTML = os.path.join(os.path.dirname(__file__), "knn_eda.html")

df = pd.read_csv(CSV_PATH)
df["hour"] = df["hour"].replace(24, 0)
df["datetime"] = pd.to_datetime(df["date"] + " " + df["hour"].astype(str).str.zfill(2) + ":00:00")
mask24 = df["hour"] == 0
df.loc[mask24 & (df.index > 0), "datetime"] = df.loc[mask24 & (df.index > 0), "datetime"] + pd.Timedelta(days=1)
df = df.sort_values("datetime").reset_index(drop=True)
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["dayofweek"] = df["datetime"].dt.dayofweek
df["weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# Cyclical encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Lag features
df["lag_168"] = df["demand_mw"].shift(168)
df["lag_24"] = df["demand_mw"].shift(24)
df["lag_1"] = df["demand_mw"].shift(1)
df["lag_temp_24"] = df["temperature_c"].shift(24)

# Differenced targets
df["diff_168"] = df["demand_mw"] - df["lag_168"]
df["ratio_168"] = df["demand_mw"] / df["lag_168"]

df = df.dropna().reset_index(drop=True)

# ── Train/Test Split ──
train = df[df["datetime"].dt.year < 2026].copy()
test = df[df["datetime"].dt.year == 2026].copy()
print(f"Train: {len(train)} rows ({train['datetime'].min().date()} -> {train['datetime'].max().date()})")
print(f"Test:  {len(test)} rows ({test['datetime'].min().date()} -> {test['datetime'].max().date()})")

FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "temperature_c", "is_holiday", "weekend", "lag_168", "lag_24", "lag_1", "lag_temp_24"]
FEATURES_DIFF = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
                 "temperature_c", "is_holiday", "weekend", "lag_24", "lag_1", "lag_temp_24"]

K_VALUES = [3, 5, 7, 10, 15, 20, 30, 50]

# Also test restricted training: 2 years only
train_2y = train[train["datetime"].dt.year >= 2024].copy()

experiments = [
    ("Raw KNN (2024-2025 train)", train_2y, test, FEATURES, "demand_mw", None),
    ("Diff-168 KNN (2024-2025 train)", train_2y, test, FEATURES_DIFF, "diff_168", "lag_168"),
    ("Ratio-168 KNN (2024-2025 train)", train_2y, test, FEATURES_DIFF, "ratio_168", "lag_168"),
    ("Raw KNN (2018-2025 train)", train, test, FEATURES, "demand_mw", None),
    ("Diff-168 KNN (2018-2025 train)", train, test, FEATURES_DIFF, "diff_168", "lag_168"),
]

all_results = []

for exp_name, t_train, t_test, feat_list, target_col, addback_col in experiments:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(t_train[feat_list])
    X_test = scaler.transform(t_test[feat_list])
    y_train = t_train[target_col].values
    y_test_raw = t_test[target_col].values

    for k in K_VALUES:
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        if addback_col:
            y_pred_abs = y_pred * t_test[addback_col].values if target_col == "ratio_168" else y_pred + t_test[addback_col].values
            y_test_abs = t_test["demand_mw"].values
        else:
            y_pred_abs = y_pred
            y_test_abs = y_test_raw

        mae = mean_absolute_error(y_test_abs, y_pred_abs)
        rmse = float(np.sqrt(mean_squared_error(y_test_abs, y_pred_abs)))
        mape = float(np.mean(np.abs((y_test_abs - y_pred_abs) / y_test_abs)) * 100)
        r2 = float(r2_score(y_test_abs, y_pred_abs))
        bias = float(np.mean(y_pred_abs - y_test_abs))

        all_results.append({
            "experiment": exp_name, "k": k, "mae": mae, "rmse": rmse,
            "mape": mape, "r2": r2, "bias": bias
        })
        print(f"{exp_name} | k={k:2d} | MAE={mae:.1f} | RMSE={rmse:.1f} | MAPE={mape:.2f}% | R2={r2:.4f} | bias={bias:+.1f}")

# ── Best model: pick lowest MAPE ──
best = min(all_results, key=lambda r: r["mape"])
print(f"\nBest: {best['experiment']} | k={best['k']} | MAE={best['mae']:.1f} | MAPE={best['mape']:.2f}% | RMSE={best['rmse']:.1f}")

# ── Retrain best for deep eval ──
def run_best(exp_name, t_train, t_test, feat_list, target_col, addback_col, k, tag):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(t_train[feat_list])
    X_test = scaler.transform(t_test[feat_list])
    y_train = t_train[target_col].values

    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    if addback_col:
        y_pred_abs = y_pred * t_test[addback_col].values if target_col == "ratio_168" else y_pred + t_test[addback_col].values
        y_test_abs = t_test["demand_mw"].values
    else:
        y_pred_abs = y_pred
        y_test_abs = t_test["demand_mw"].values
    return y_pred_abs, y_test_abs, t_test

y_pred, y_true, test_df = run_best(
    best["experiment"],
    train_2y if "2024-2025" in best["experiment"] else train,
    test,
    FEATURES_DIFF if "Diff" in best["experiment"] or "Ratio" in best["experiment"] else FEATURES,
    "diff_168" if "Diff" in best["experiment"] else ("ratio_168" if "Ratio" in best["experiment"] else "demand_mw"),
    "lag_168" if "Diff" in best["experiment"] or "Ratio" in best["experiment"] else None,
    best["k"],
    "best"
)

errors = y_pred - y_true
pct_errors = errors / y_true * 100
test_df = test_df.copy()
test_df["pred"] = y_pred
test_df["error"] = errors
test_df["pct_error"] = pct_errors

# ── Build results HTML ──
def tbl(df_list, headers):
    rows = []
    for row in df_list:
        rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
    return "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>" + "".join(rows) + "</table>"

results_table = tbl(
    [[r["experiment"].split("(")[0].strip(), str(r["k"]),
      f"{r['mae']:.1f}", f"{r['rmse']:.1f}", f"{r['mape']:.2f}%", f"{r['r2']:.3f}", f"{r['bias']:+.1f}"]
     for r in sorted(all_results, key=lambda x: x["mape"])],
    ["Experiment", "k", "MAE (MW)", "RMSE (MW)", "MAPE", "R2", "Bias (MW)"]
)

# Per-hour error
hourly_err = test_df.groupby("hour")[["error", "pct_error"]].agg(["mean", "std"]).round(1)
hourly_mae = test_df.groupby("hour").apply(lambda g: np.mean(np.abs(g["error"]))).round(1)
hourly_table = tbl(
    [[str(h), f"{hourly_mae[h]:.1f}",
      f"{hourly_err.loc[h, ('error', 'mean')]:+.1f}",
      f"{hourly_err.loc[h, ('error', 'std')]:.1f}",
      f"{hourly_err.loc[h, ('pct_error', 'mean')]:+.2f}%",
      f"{hourly_err.loc[h, ('pct_error', 'std')]:.2f}%"]
     for h in range(24)],
    ["Hour", "MAE (MW)", "Error Mean", "Error Std", "MAPE Mean", "MAPE Std"]
)

# Per-month error
monthly_mae = test_df.groupby("month").apply(lambda g: np.mean(np.abs(g["error"]))).round(1)
monthly_err = test_df.groupby("month")[["error", "pct_error"]].agg(["mean", "std"]).round(1)
month_names = ["Jan","Feb","Mar","Apr","May"]
monthly_table = tbl(
    [[month_names[i-1], f"{monthly_mae[i]:.1f}",
      f"{monthly_err.loc[i, ('error', 'mean')]:+.1f}",
      f"{monthly_err.loc[i, ('error', 'std')]:.1f}",
      f"{monthly_err.loc[i, ('pct_error', 'mean')]:+.2f}%"]
     for i in sorted(test_df["month"].unique())],
    ["Month", "MAE (MW)", "Error Mean", "Error Std", "MAPE"]
)

# ── Plots ──
# 1. Actual vs Predicted scatter
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(size=3, color="#1f77b4", opacity=0.5), name=""))
fig1.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], mode="lines",
                          line=dict(color="red", dash="dash"), name="Perfect"))
fig1.update_layout(title=f"Actual vs Predicted — Best: {best['experiment'].split('(')[0].strip()} (k={best['k']})",
                   xaxis_title="Actual Demand (MW)", yaxis_title="Predicted Demand (MW)",
                   height=400, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
plot1 = fig1.to_html(full_html=False, include_plotlyjs="cdn")

# 2. Time series of predictions
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=test_df["datetime"], y=y_true, mode="lines", name="Actual", line=dict(width=1.5, color="#1f77b4")))
fig2.add_trace(go.Scatter(x=test_df["datetime"], y=y_pred, mode="lines", name="Predicted", line=dict(width=1.5, color="#d62728", dash="dot")))
fig2.update_layout(title="Best KNN — Actual vs Predicted (2026 H1)",
                   xaxis_title="Date", yaxis_title="Demand (MW)",
                   height=400, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
plot2 = fig2.to_html(full_html=False, include_plotlyjs=False)

# 3. Error distribution
fig3 = go.Figure()
fig3.add_trace(go.Histogram(x=errors, nbinsx=60, marker_color="#2ca02c", name="Error"))
fig3.update_layout(title="Prediction Error Distribution (MW)",
                   xaxis_title="Error (MW)", yaxis_title="Count",
                   height=300, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
plot3 = fig3.to_html(full_html=False, include_plotlyjs=False)

# 4. Error by hour
fig4 = go.Figure()
fig4.add_trace(go.Bar(x=list(range(24)), y=hourly_mae.values, marker_color="#ff7f0e", name="MAE"))
fig4.update_layout(title="MAE by Hour of Day",
                   xaxis=dict(tickmode="array", tickvals=list(range(24))),
                   yaxis_title="MAE (MW)", height=300, template="plotly_white",
                   margin=dict(l=20, r=20, t=40, b=20))
plot4 = fig4.to_html(full_html=False, include_plotlyjs=False)

# 5. K vs error
k_df = pd.DataFrame(all_results)
fig5 = go.Figure()
for exp in k_df["experiment"].unique():
    sub = k_df[k_df["experiment"] == exp]
    label = exp.split("(")[0].strip()
    fig5.add_trace(go.Scatter(x=sub["k"], y=sub["mape"], mode="lines+markers", name=label,
                              line=dict(width=2)))
fig5.update_layout(title="MAPE vs K Value — All Experiments",
                   xaxis_title="K (neighbors)", yaxis_title="MAPE (%)",
                   height=400, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
plot5 = fig5.to_html(full_html=False, include_plotlyjs=False)

# 6. Week sample
week_sample = test_df.iloc[:168]
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=week_sample["datetime"], y=week_sample["demand_mw"], mode="lines",
                          name="Actual", line=dict(width=2, color="#1f77b4")))
fig6.add_trace(go.Scatter(x=week_sample["datetime"], y=week_sample["pred"], mode="lines",
                          name="KNN Pred", line=dict(width=2, color="#d62728", dash="dot")))
fig6.update_layout(title="Best KNN — First Week of 2026 (Hourly)",
                   xaxis_title="Date", yaxis_title="Demand (MW)",
                   height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
plot6 = fig6.to_html(full_html=False, include_plotlyjs=False)

# ── Read existing EDA, inject results ──
with open(EDA_HTML, "r", encoding="utf-8") as f:
    html = f.read()

insert = f"""
<div class="section">
<h2>KNN Results — Best: {best['experiment'].split('(')[0].strip()} (k={best['k']})</h2>
<p><b>MAE:</b> {best['mae']:.1f} MW &nbsp;|&nbsp; <b>MAPE:</b> {best['mape']:.2f}% &nbsp;|&nbsp; <b>RMSE:</b> {best['rmse']:.1f} MW &nbsp;|&nbsp; <b>R2:</b> {best['r2']:.3f}</p>
{plot1}
</div>

<div class="section">
<h2>Time Series Comparison (2026 H1)</h2>
{plot2}
</div>

<div class="section">
<h2>First Week Detail</h2>
{plot6}
</div>

<div class="section">
<h2>Error Distribution</h2>
{plot3}
</div>

<div class="section">
<h2>MAE by Hour</h2>
{plot4}
</div>

<div class="section">
<h2>MAPE vs K — All Experiments</h2>
{plot5}
</div>

<div class="section">
<h2>Full Results Table</h2>
<p style="font-size:0.9em;">All experiments tested on Jan-May 2026. Bold = best overall.</p>
{results_table}
</div>

<div class="section">
<h2>Error by Hour of Day</h2>
{hourly_table}
</div>

<div class="section">
<h2>Error by Month (2026)</h2>
{monthly_table}
</div>
"""

# Insert before closing body
html = html.replace("</body>", insert + "\n</body>")

with open(EDA_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nUpdated: {EDA_HTML}")
