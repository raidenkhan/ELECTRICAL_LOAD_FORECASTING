import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import timedelta
import json
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "dl_forecast", "ecg_demand_2018_2026.csv")
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "eda", "knn_eda.html")

df = pd.read_csv(CSV_PATH)
df["hour"] = df["hour"].replace(24, 0)
df["datetime"] = pd.to_datetime(df["date"] + " " + df["hour"].astype(str).str.zfill(2) + ":00:00")
mask24 = df["hour"] == 0
df.loc[mask24 & (df.index > 0), "datetime"] = df.loc[mask24 & (df.index > 0), "datetime"] + pd.Timedelta(days=1)
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["dayofweek"] = df["datetime"].dt.dayofweek
df["weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

sections = []

def section(title, html):
    sections.append(f"""<div class="section"><h2>{title}</h2>{html}</div>""")

# ── 1. Dataset Stats ──
stats_html = df[["demand_mw", "temperature_c"]].describe().round(1).to_html()
stats_html += "<br><b>Shape:</b> %d rows, %d columns<br>" % df.shape
stats_html += ("<b>Date range:</b> %s → %s<br>" % (df["datetime"].min().strftime("%Y-%m-%d"), df["datetime"].max().strftime("%Y-%m-%d")))
stats_html += "<b>Holidays:</b> %d / %d (%.1f%%)<br>" % (df["is_holiday"].sum(), len(df), df["is_holiday"].mean()*100)
section("1. Dataset Overview", stats_html)

# ── 2. Full Time Series ──
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["datetime"], y=df["demand_mw"], mode="lines", name="Demand", line=dict(width=1, color="#1f77b4")))
fig.update_layout(title="Demand (MW) — Full Time Series", xaxis_title="Date", yaxis_title="MW",
                  height=400, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
section("2. Time Series Overview", fig.to_html(full_html=False, include_plotlyjs="cdn"))

# ── 3. Year-by-Year Comparison ──
fig = go.Figure()
colors = px.colors.qualitative.Set2
for i, yr in enumerate(sorted(df["year"].unique())):
    sub = df[df["year"] == yr].copy()
    if len(sub) < 24: continue
    sub["doy"] = sub["datetime"].dt.dayofyear
    fig.add_trace(go.Scatter(x=sub["doy"], y=sub["demand_mw"], mode="lines",
                             name=str(yr), line=dict(width=1.2, color=colors[i % len(colors)])))
fig.update_layout(title="Year-over-Year Demand (by Day of Year)", xaxis_title="Day of Year", yaxis_title="MW",
                  height=450, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
section("3. Year-over-Year", fig.to_html(full_html=False, include_plotlyjs=False))

# ── 4. Hourly Pattern (24h boxplot) ──
fig = go.Figure()
for h in range(24):
    vals = df[df["hour"] == h]["demand_mw"]
    fig.add_trace(go.Box(y=vals, name=str(h), boxmean="sd", line=dict(width=1),
                         marker_color="#2ca02c", whiskerwidth=0.5))
fig.update_layout(title="Demand by Hour — KNN Feature: hour", xaxis_title="Hour", yaxis_title="MW",
                  height=400, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
section("4. Hourly Pattern (Box Plot)", fig.to_html(full_html=False, include_plotlyjs=False))

# ── 5. Day-of-Week Pattern ──
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig = go.Figure()
for d in range(7):
    vals = df[df["dayofweek"] == d]["demand_mw"]
    fig.add_trace(go.Box(y=vals, name=dow_labels[d], boxmean="sd", line=dict(width=1),
                         marker_color="#ff7f0e", whiskerwidth=0.5))
fig.update_layout(title="Demand by Day of Week — KNN Feature: dayofweek", xaxis_title="Day", yaxis_title="MW",
                  height=400, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
section("5. Day-of-Week Pattern", fig.to_html(full_html=False, include_plotlyjs=False))

# ── 6. Monthly Seasonality ──
monthly = df.groupby("month")["demand_mw"].agg(["mean", "std", "min", "max"]).reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["mean"], mode="lines+markers", name="Mean",
                         line=dict(width=2, color="#d62728"), marker=dict(size=8)))
fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["min"], mode="lines", name="Min",
                         line=dict(width=1, dash="dash", color="#9467bd")))
fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["max"], mode="lines", name="Max",
                         line=dict(width=1, dash="dash", color="#9467bd")))
fig.update_layout(title="Monthly Seasonality — KNN Feature: month", xaxis=dict(tickmode="array", tickvals=list(range(1,13)), ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]),
                  yaxis_title="MW", height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
section("6. Monthly Seasonality", fig.to_html(full_html=False, include_plotlyjs=False))

# ── 7. Temperature vs Demand ──
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["temperature_c"], y=df["demand_mw"], mode="markers",
                         marker=dict(size=3, color=df["hour"], colorscale="Viridis", showscale=True,
                                     colorbar=dict(title="Hour")), name=""))
fig.update_layout(title="Temperature vs Demand — KNN Feature: temperature_c (colored by hour)",
                  xaxis_title="Temperature (°C)", yaxis_title="Demand (MW)",
                  height=450, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
section("7. Temperature vs Demand", fig.to_html(full_html=False, include_plotlyjs=False))

# ── 8. Holiday vs Non-Holiday ──
fig = go.Figure()
fig.add_trace(go.Box(y=df[df["is_holiday"]==0]["demand_mw"], name="Non-Holiday", boxmean="sd", marker_color="#1f77b4"))
fig.add_trace(go.Box(y=df[df["is_holiday"]==1]["demand_mw"], name="Holiday", boxmean="sd", marker_color="#d62728"))
fig.update_layout(title="Holiday vs Non-Holiday Demand — KNN Feature: is_holiday", yaxis_title="MW",
                  height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
section("8. Holiday Effect", fig.to_html(full_html=False, include_plotlyjs=False))

# ── 9. Weekend vs Weekday ──
fig = go.Figure()
fig.add_trace(go.Box(y=df[df["weekend"]==0]["demand_mw"], name="Weekday", boxmean="sd", marker_color="#1f77b4"))
fig.add_trace(go.Box(y=df[df["weekend"]==1]["demand_mw"], name="Weekend", boxmean="sd", marker_color="#ff7f0e"))
fig.update_layout(title="Weekday vs Weekend Demand", yaxis_title="MW",
                  height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
section("9. Weekend Effect", fig.to_html(full_html=False, include_plotlyjs=False))

# ── 10. Demand Distribution ──
fig = make_subplots(rows=1, cols=2, subplot_titles=("Demand Distribution", "Log Distribution"))
fig.add_trace(go.Histogram(x=df["demand_mw"], nbinsx=80, marker_color="#1f77b4", name="MW"), row=1, col=1)
fig.add_trace(go.Histogram(x=np.log1p(df["demand_mw"]), nbinsx=80, marker_color="#2ca02c", name="log(MW)"), row=1, col=2)
fig.update_layout(title="Demand Distribution", height=350, template="plotly_white", showlegend=False,
                  margin=dict(l=20, r=20, t=40, b=20))
section("10. Distribution", fig.to_html(full_html=False, include_plotlyjs=False))

# ── 11. Autocorrelation (for KNN lag features) ──
daily = df.set_index("datetime")["demand_mw"].resample("D").mean()
acf = [daily.autocorr(lag=i) for i in range(1, 365)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, 365)), y=acf, mode="lines", name="ACF", line=dict(color="#1f77b4", width=2)))
fig.add_trace(go.Scatter(x=[0, 365], y=[0.05, 0.05], mode="lines", line=dict(dash="dash", color="gray", width=1), name="95% CI"))
fig.add_trace(go.Scatter(x=[0, 365], y=[-0.05, -0.05], mode="lines", line=dict(dash="dash", color="gray", width=1), showlegend=False))
fig.update_layout(title="Autocorrelation (Daily Avg) — Choose KNN Lags", xaxis_title="Lag (days)", yaxis_title="Autocorrelation",
                  height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
section("11. Autocorrelation (Daily)", fig.to_html(full_html=False, include_plotlyjs=False))

# ── 12. Correlation Matrix ──
features = df[["demand_mw", "temperature_c", "hour", "dayofweek", "month", "weekend", "is_holiday"]].copy()
# Add lag features
for lag in [1, 2, 3, 24, 48, 168]:
    features["lag_%d" % lag] = df["demand_mw"].shift(lag)
features = features.dropna()
corr = features.corr()
labels = list(corr.columns)
z = corr.values
fig = go.Figure(data=go.Heatmap(z=z, x=labels, y=labels, colorscale="RdBu_r", zmin=-1, zmax=1,
                                 text=np.round(z, 2), texttemplate="%{text}", textfont=dict(size=9)))
fig.update_layout(title="Correlation Matrix (with Lags)", height=600, width=700,
                  template="plotly_white", margin=dict(l=60, r=20, t=40, b=60))
section("12. Correlation Matrix (with Lags)", fig.to_html(full_html=False, include_plotlyjs=False))

# ── 13. Growth Trend ──
daily_mean = df.groupby("date")["demand_mw"].mean().reset_index()
daily_mean["date"] = pd.to_datetime(daily_mean["date"])
z = np.polyfit(range(len(daily_mean)), daily_mean["demand_mw"], 1)
p = np.poly1d(z)
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily_mean["date"], y=daily_mean["demand_mw"], mode="markers",
                         marker=dict(size=3, color="#1f77b4", opacity=0.6), name="Daily Mean"))
fig.add_trace(go.Scatter(x=daily_mean["date"], y=p(range(len(daily_mean))), mode="lines",
                         line=dict(color="#d62728", width=3), name="Trend (+%.0f MW/yr)" % (z[0]*365)))
fig.update_layout(title="Demand Growth Trend — Critical for KNN (non-stationary!)",
                  xaxis_title="Date", yaxis_title="Mean Daily Demand (MW)",
                  height=400, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
section("13. Growth Trend", fig.to_html(full_html=False, include_plotlyjs=False))

# ── 14. Yearly Hourly Profiles (overlay) ──
fig = go.Figure()
for yr in sorted(df["year"].unique()):
    sub = df[df["year"] == yr].groupby("hour")["demand_mw"].mean()
    fig.add_trace(go.Scatter(x=sub.index, y=sub.values, mode="lines", name=str(yr), line=dict(width=1.5)))
fig.update_layout(title="Average Hourly Profile by Year — KNN needs recent data",
                  xaxis_title="Hour", yaxis_title="Mean Demand (MW)",
                  height=400, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
section("14. Hourly Profile Evolution", fig.to_html(full_html=False, include_plotlyjs=False))

# ── Assemble HTML ──
dmin = df["datetime"].min().strftime("%Y-%m-%d")
dmax = df["datetime"].max().strftime("%Y-%m-%d")
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GRIDCo Demand EDA — KNN Preparation</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
h1 {{ text-align: center; color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
h2 {{ color: #283593; margin-top: 30px; background: #e8eaf6; padding: 8px 15px; border-radius: 4px; }}
.section {{ background: white; border-radius: 8px; padding: 15px 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ padding: 6px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
th {{ background: #1a237e; color: white; }} tr:hover {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h1>GRIDCo Load Demand — EDA for K-Nearest Neighbors</h1>
<p style="text-align:center;color:#666;">Dataset: ecg_demand_2018_2026.csv | {len(df)} rows | {dmin} -&gt; {dmax}</p>"""

html += "\n".join(sections)

html += """
<div class="section">
<h2>KNN Takeaways</h2>
<ul>
<li><b>Strong seasonality</b> (hour ×24, dayofweek, month) — these are critical KNN features</li>
<li><b>Temperature correlation</b> — non-linear U-shape: demand rises at both low and high temps</li>
<li><b>Non-stationary trend</b> — demand nearly doubled (1.7k → 3.3k MW). KNN <b>must</b> weight recent data higher or use ratio/differencing features</li>
<li><b>Hourly profiles shift</b> — the shape changes year-to-year; restrict KNN search to recent years or use time-weighted distance</li>
<li><b>Strong autocorrelation</b> at lag 1, 7, 365 days — lag features are very informative</li>
<li><b>Holiday/weekend effect</b> is real but small (~5% lower) — include as binary features</li>
<li><b>Consider differencing</b> (demand<sub>t</sub> - demand<sub>t-168</sub>) to remove trend+seasonality before KNN</li>
</ul>
</div>
</body>
</html>"""

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(html)

print("Written to:", OUTPUT)
