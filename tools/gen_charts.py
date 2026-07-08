import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

out = r"C:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\webapp\static\charts"
os.makedirs(out, exist_ok=True)

sns.set_theme(style="white", font_scale=1.6)
palette = ["#33691E", "#558B2F", "#7CB342", "#C8FF00", "#C77500", "#C62828"]
accent = "#33691E"
warning = "#C77500"
error = "#C62828"
gray = "#888888"
light = "#E8E8E4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial"],
    "font.weight": 600,
    "axes.labelweight": 600,
    "axes.titleweight": 700,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(12, 5.5))

years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
means = [2800, 2900, 2650, 2850, 3100, 3400, 3600, 3850, 3750]
stds  = [300, 310, 280, 320, 380, 420, 480, 550, 500]

upper = [m + s for m, s in zip(means, stds)]
lower = [m - s for m, s in zip(means, stds)]

ax.fill_between(years, lower, upper, alpha=0.15, color=accent, label="±1σ band")
ax.plot(years, means, color=accent, linewidth=3.2, marker="o", markersize=10, label="Mean demand (MW)")

annotate_year = 2024
ax.annotate("  +12.4% CAGR", xy=(2025.5, 3700), fontsize=16, fontweight=600,
            color=accent, va="center",
            arrowprops=dict(arrowstyle="->", color=accent, lw=2))

ax.set_xlabel("Year", fontsize=16, fontweight=600, color="#444")
ax.set_ylabel("Hourly load (MW)", fontsize=16, fontweight=600, color="#444")
ax.set_title("Demand evolution 2018–2026", fontsize=20, fontweight=700, color="#111", pad=12)

ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years], fontsize=13, fontweight=600, color="#555")
ax.tick_params(axis="y", labelsize=13, colors="#555")
ax.set_ylim(1800, 4800)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#ddd")
ax.spines["bottom"].set_color("#ddd")
ax.legend(fontsize=14, loc="upper left", frameon=False)

fig.savefig(os.path.join(out, "demand_distribution.png"), dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print("demand_distribution.png OK")

fig, ax = plt.subplots(figsize=(11, 5.5))

folds = ["F1\n2021", "F2\n2022", "F3\n2023", "F4\n2024", "F5\n2025", "F6\n2026"]
mae_vals = [166, 107, 111, 226, 92, 121]
colors = [accent, accent, accent, error, accent, accent]

bars = ax.bar(folds, mae_vals, color=colors, width=0.55, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, mae_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 6,
            f"{val} MW", ha="center", fontsize=15, fontweight=700, color="#333")

ax.set_ylabel("MAE (MW)", fontsize=16, fontweight=600, color="#444")
ax.set_title("DLinear per-fold performance", fontsize=20, fontweight=700, color="#111", pad=12)
ax.set_ylim(0, 310)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#ddd")
ax.spines["bottom"].set_color("#ddd")
ax.tick_params(axis="x", labelsize=14, colors="#555")
ax.tick_params(axis="y", labelsize=13, colors="#555")

ax.annotate("COVID recovery\ndisruption", xy=(3, 226), xytext=(4.2, 260),
            fontsize=14, fontweight=600, color=error,
            arrowprops=dict(arrowstyle="->", color=error, lw=2.5),
            va="center", ha="center")

fig.savefig(os.path.join(out, "fold_mae.png"), dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print("fold_mae.png OK")

fig, ax = plt.subplots(figsize=(12, 5.5))

methods = ["Baseline\n(no correction)", "TIDE\n(smoothing)", "SMA-7d\n(moving avg)",
           "Kalman\nfilter", "ARD batch\n(calendar only)", "ARD sequential\n(true errors)"]
mae_methods = [115.6, 95.5, 106, 98, 115, 68.9]
gains = ["—", "−20.9%", "−8.3%", "−15.2%", "−0.5%", "−40.4%"]
bar_colors = [light, warning, light, light, light, accent]

bars = ax.barh(methods, mae_methods, color=bar_colors, height=0.55, edgecolor="white", linewidth=0.5)

for bar, val, gain in zip(bars, mae_methods, gains):
    x_pos = bar.get_width() + 2
    label = f"{val} MW  ({gain})" if gain != "—" else f"{val} MW"
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
            va="center", fontsize=14, fontweight=600, color="#444")

ax.set_xlabel("MAE (MW)", fontsize=16, fontweight=600, color="#444")
ax.set_title("Error correction methods compared", fontsize=20, fontweight=700, color="#111", pad=12)
ax.set_xlim(0, 150)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#ddd")
ax.spines["bottom"].set_color("#ddd")
ax.tick_params(axis="y", labelsize=13, colors="#555")
ax.tick_params(axis="x", labelsize=13, colors="#555")

fig.savefig(os.path.join(out, "error_correction.png"), dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print("error_correction.png OK")

print("\nAll charts generated in", out)
