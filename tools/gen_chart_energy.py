import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

out = r"C:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\webapp\static\charts"
os.makedirs(out, exist_ok=True)

sns.set_theme(style="white", font_scale=1.5)
accent = "#33691E"
warning = "#C77500"
error = "#C62828"
gray = "#888888"
light = "#E8E8E4"
dark = "#444444"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial"],
    "font.weight": 600,
    "axes.labelweight": 600,
    "axes.titleweight": 700,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1]})

models = ["Simple\nTrend", "KNN\n(similar-day)", "Classical\nDecomp.", "DLinear\n(36K params)", "CNN\n(1.2M params)", "LSTM\n(840K params)", "Transformer\n(2.1M params)"]
mae_vals = [113, 141, 105, 91, 97, 102, 109]
energy_vals = [0, 0, 0, 0.003, 0.02, 0.03, 0.05]
energy_labels = ["0", "0", "0", "0.003", "0.02", "0.03", "0.05"]
bar_colors_mae = [light, light, light, accent, light, light, light]

bars1 = ax1.bar(models, mae_vals, color=bar_colors_mae, width=0.6, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars1, mae_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f"{val} MW", ha="center", fontsize=13, fontweight=700, color="#333")

ax1.set_ylabel("MAE (MW)  ↓ lower is better", fontsize=14, fontweight=600, color=dark)
ax1.set_title("Forecast Accuracy", fontsize=17, fontweight=700, color="#111", pad=10)
ax1.set_ylim(0, 185)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_color("#ddd")
ax1.spines["bottom"].set_color("#ddd")
ax1.tick_params(axis="x", labelsize=10, colors=dark)
ax1.tick_params(axis="y", labelsize=12, colors=gray)

bar_colors_energy = [light, light, light, accent, warning, warning, error]
bars2 = ax2.bar(models, energy_vals, color=bar_colors_energy, width=0.6, edgecolor="white", linewidth=0.5)

for bar, val, lbl in zip(bars2, energy_vals, energy_labels):
    y_pos = bar.get_height() + 0.001 if val > 0 else 0.001
    ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
             f"{lbl} kWh", ha="center", fontsize=13, fontweight=700, color="#333" if val > 0 else "#999")

ax2.set_ylabel("Training energy (kWh)  ↓ lower is better", fontsize=14, fontweight=600, color=dark)
ax2.set_title("Energy Cost per Training Fold", fontsize=17, fontweight=700, color="#111", pad=10)
ax2.set_ylim(0, 0.07)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_color("#ddd")
ax2.spines["bottom"].set_color("#ddd")
ax2.tick_params(axis="x", labelsize=10, colors=dark)
ax2.tick_params(axis="y", labelsize=12, colors=gray)

plt.tight_layout(pad=2)
fig.savefig(os.path.join(out, "energy_efficiency.png"), dpi=150, bbox_inches="tight", pad_inches=0.4)
plt.close(fig)
print("energy_efficiency.png OK")
