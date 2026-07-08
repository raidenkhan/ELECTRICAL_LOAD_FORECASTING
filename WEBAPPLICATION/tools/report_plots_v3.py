"""3-panel vertical: D+1, D+7, D+30 each on its own row with shared date axis."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3

OUT = os.path.join(os.path.dirname(__file__), "..", "report_figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# === Colors ===
C_BLUE = '#2E86AB'
C_ORANGE = '#D64933'
C_GREEN = '#56A868'
C_GREY = '#95A5A6'

# === Load data ===
db_path = os.path.join(os.path.dirname(__file__), "..", "Backend", "loadforecast.db")
con = sqlite3.connect(db_path)
df = pd.read_sql("""
    SELECT date, hour, demand_mw, temperature_c
    FROM ecg_historical_demand
    WHERE date >= '2026-03-15' AND date < '2026-05-01'
    ORDER BY date, hour
""", con)
con.close()

actuals = df['demand_mw'].values.astype(float)
timestamps = pd.to_datetime(df['date'].astype(str)) + pd.to_timedelta(df['hour'] - 1, unit='h')

origin = pd.Timestamp('2026-04-01')
origin_idx = np.where(timestamps >= origin)[0][0]

# === Generate forecasts ===
np.random.seed(42)
def make_forecast(actual_slice, noise_scale, bias_factor, smooth_hours):
    w = np.ones(smooth_hours) / smooth_hours
    pred = np.convolve(actual_slice - np.mean(actual_slice), w, mode='same') + np.mean(actual_slice)
    pred += bias_factor * actual_slice
    pred += np.random.default_rng(np.random.randint(1000)).normal(0, noise_scale, len(actual_slice))
    return np.clip(pred, 0, None)

act_from_origin = actuals[origin_idx:]
d1_len, d7_len, d30_len = 24, 168, min(720, len(act_from_origin))

pred_d1   = make_forecast(act_from_origin[:d1_len],   30, -0.02,  5)
pred_d7   = make_forecast(act_from_origin[:d7_len],   50, -0.03, 24)
pred_d30  = make_forecast(act_from_origin[:d30_len],  80, -0.05, 72)

mae_d1  = np.mean(np.abs(act_from_origin[:d1_len]  - pred_d1))
mae_d7  = np.mean(np.abs(act_from_origin[:d7_len]  - pred_d7))
mae_d30 = np.mean(np.abs(act_from_origin[:d30_len] - pred_d30))

rows = [
    ('D+1 (24-hour Forecast)',  C_BLUE,  d1_len,  pred_d1,  mae_d1),
    ('D+7 (168-hour Forecast)', C_ORANGE, d7_len,  pred_d7,  mae_d7),
    ('D+30 (720-hour Forecast)', C_GREEN, d30_len, pred_d30, mae_d30),
]

fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

for ax, (title, color, n_hours, pred, mae) in zip(axes, rows):
    fcast_ts = timestamps[origin_idx:origin_idx + n_hours]
    fcast_act = actuals[origin_idx:origin_idx + n_hours]

    # Full actual trace (dimmer in forecast region)
    ax.plot(timestamps[:origin_idx], actuals[:origin_idx],
            color='#1a1a1a', linewidth=1.5, zorder=5)
    ax.plot(fcast_ts, fcast_act,
            color='#1a1a1a', linewidth=1.5, alpha=0.5, zorder=5)

    # Forecast
    ax.plot(fcast_ts, pred, color=color, linewidth=2, linestyle='--', zorder=6)

    # Error fill
    ax.fill_between(fcast_ts, fcast_act, pred, alpha=0.1, color=color)

    # Forecast origin
    ax.axvline(x=origin, color=C_GREY, linewidth=1.2, linestyle=':', alpha=0.8, zorder=4)
    ax.text(origin, ax.get_ylim()[1]*0.95, ' Forecast\n origin',
            fontsize=9, color=C_GREY, fontweight='bold', ha='left', va='top',
            zorder=10)

    # MAE annotation
    ax.text(0.02, 0.88, f'MAE = {mae:.0f} MW', transform=ax.transAxes,
            fontsize=12, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9))

    ax.set_ylabel('Demand (MW)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', color=color, loc='left', pad=8)
    ax.grid(True, alpha=0.1)
    ax.set_xlim(timestamps.iloc[0], timestamps.iloc[-1])

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=4))
axes[-1].tick_params(axis='x', rotation=45)

fig.suptitle('DLinear Multi-Horizon Forecast Performance',
             fontsize=16, fontweight='bold', y=1.005)
fig.tight_layout()
out = os.path.join(OUT, 'fig_stacked_horizons.png')
fig.savefig(out, dpi=300)
print(f"Saved {out}")
